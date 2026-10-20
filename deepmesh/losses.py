"""Loss functions of Eq. (11) and Eq. (12).

L_smooth and L_surf follow pytorch3d (mesh_laplacian_smoothing with uniform
weights, chamfer_distance with squared distances and mean reductions); they are
written here in plain PyTorch so that the code also runs on CPU and AMD GPUs.
"""
import math

import torch
import torch.nn.functional as F

from deepmesh.geometry import warp
from deepmesh.rasterizer import plane_probability, selection_threshold, to_plane


def laplacian_smoothing(verts, edges):
    """Uniform Laplacian smoothing loss (Eq. 3). verts (B, N, 3), edges (E, 2) shared topology."""
    a, b = edges[:, 0], edges[:, 1]
    nbr = torch.zeros_like(verts)
    nbr.index_add_(1, a, verts[:, b])
    nbr.index_add_(1, b, verts[:, a])
    deg = torch.bincount(edges.flatten(), minlength=verts.shape[1]).clamp(min=1).to(verts.dtype)
    lap = nbr / deg[None, :, None] - verts
    return lap.norm(dim=-1).mean()


@torch.no_grad()
def _nearest(x, y, chunk=2048):
    """Index of the nearest point of y for every point of x. x (B, N, 3), y (B, M, 3)."""
    return torch.cat([torch.cdist(x[:, s:s + chunk], y).argmin(-1) for s in range(0, x.shape[1], chunk)], 1)


def chamfer_distance(x, y):
    """Symmetric Chamfer distance with squared L2 (Eq. 4), batch-averaged.

    Nearest neighbours are found without gradient in chunks (memory-friendly for
    ~20k vertices) and the distance is then recomputed for the matched pairs, which
    gives exactly the gradient of the min.
    """
    def one_way(p, q):
        nn = _nearest(p, q)
        matched = torch.gather(q, 1, nn[..., None].expand(-1, -1, 3))
        return ((p - matched) ** 2).sum(-1).mean(1)

    return (one_way(x, y) + one_way(y, x)).mean()


def huber_regularization(field, eps=0.01):
    """Huber-like smoothness of a dense field (Eq. 5), as implemented by the authors."""
    B, _, D, H, W = field.shape
    dx = field[..., 1:] - field[..., :-1]
    dy = field[..., 1:, :] - field[..., :-1, :]
    dz = field[:, :, 1:] - field[:, :, :-1]
    err = ((dx ** 2).sum() / W + (dy ** 2).sum() / H + (dz ** 2).sum() / D) / B
    return torch.sqrt(eps + err)


def similarity_loss(sa_ed, sa_t, field):
    """Eq. (8): MSE between the ED SAX stack and the frame-t stack warped by Phi_{0->t}."""
    return F.mse_loss(warp(sa_t.unsqueeze(1), field), sa_ed.unsqueeze(1))


def _generalized_mean(x, dim, p=-7):
    """Soft minimum used by the weighted Hausdorff distance (-> min as p -> -inf)."""
    return torch.mean((x + 1e-6) ** p, dim) ** (1.0 / p)


def weighted_hausdorff(points, prob, gt_points, max_dist):
    """Weighted Hausdorff distance (Ribera et al., 2019) between a soft point set
    (points (N, 2) with probabilities prob (N,)) and a ground-truth point set (M, 2)."""
    d = torch.sqrt(((points[:, None] - gt_points[None]) ** 2).sum(-1) + 1e-8)  # (N, M)
    term1 = (prob * d.min(1)[0]).sum() / (prob.sum() + 1e-6)
    p = prob[:, None]
    term2 = _generalized_mean((1 - p) * max_dist + p * d, dim=0).mean()
    return term1 + term2


def plane_shape_loss(points, k, z, gt_contour, tau):
    """Shape loss on one view plane for a batch.

    points (B, N, 2), k (B, N): output of to_plane; z: index of the plane;
    gt_contour (B, H, W) binary contour. Empty cases follow the reference code.
    """
    H, W = gt_contour.shape[-2:]
    max_dist = math.sqrt(H ** 2 + W ** 2)
    prob = plane_probability(k, z, tau)
    thr = selection_threshold(tau)
    losses = []
    for b in range(points.shape[0]):
        keep = prob[b] > thr
        pts, pr = points[b][keep], prob[b][keep]
        gt = torch.nonzero(gt_contour[b]).to(points.dtype)
        empty_pred = pr.sum() < 1e-3
        if len(gt) == 0:
            losses.append(points.new_tensor(0.0 if empty_pred else max_dist))
        elif empty_pred:
            losses.append(points.new_tensor(max_dist))
        else:
            losses.append(weighted_hausdorff(pts, pr, gt, max_dist))
    return torch.stack(losses).mean()


def shape_loss(verts, affines, contours, sa_slices, tau, views=("sa", "2ch", "4ch")):
    """L_shape of Eq. (10): mean over SAX slices + 2CH + 4CH.

    contours: dict with 'sa' (B, H, W, S) for the slice indices `sa_slices`, and
    '2ch'/'4ch' (B, H, W). LAX images have a single slice, so their plane is k = 0.
    """
    H, W = contours["sa"].shape[1:3]
    total, parts = 0.0, {}
    if "sa" in views:
        pts, k = to_plane(verts, affines["sa_inv"], H, W)
        sa = [plane_shape_loss(pts, k, z, contours["sa"][..., s], tau) for s, z in enumerate(sa_slices)]
        parts["sa"] = torch.stack(sa).mean()
    for v in ("2ch", "4ch"):
        if v in views:
            pts, k = to_plane(verts, affines[f"{v}_inv"], H, W)
            parts[v] = plane_shape_loss(pts, k, 0, contours[v], tau)
    for value in parts.values():
        total = total + value
    return total, parts
