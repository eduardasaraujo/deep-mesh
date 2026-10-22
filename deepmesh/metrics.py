"""Evaluation metrics of Sec. IV-A2: surface distance, Hausdorff distance and
boundary F-score between mesh contours and ground-truth contours, plus the
vertex-wise error that only synthetic data allows."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from skimage.morphology import binary_dilation, disk


def mesh_contour_points(verts, affine_inv, z, band, height, width):
    """Pixels where the mesh crosses plane z: vertices with |k - z| < band, rounded."""
    vox = verts @ affine_inv[:3, :3].T + affine_inv[:3, 3]
    vox = vox[np.abs(vox[:, 2] - z) < band]
    i = np.clip(np.round(vox[:, 0]), 0, height - 1).astype(int)
    j = np.clip(np.round(vox[:, 1]), 0, width - 1).astype(int)
    return np.stack([i, j], 1)


def points_to_mask(points, height, width):
    mask = np.zeros((height, width), bool)
    mask[points[:, 0], points[:, 1]] = True
    return mask


def contour_distances(pts_a, pts_b, spacing):
    """Mean contour distance and Hausdorff distance in mm (None if a set is empty)."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return None, None
    d = cdist(pts_a, pts_b)
    mcd = 0.5 * (d.min(0).mean() + d.min(1).mean()) * spacing
    hd = max(d.min(0).max(), d.min(1).max()) * spacing
    return mcd, hd


def boundary_fscore(pred, gt, bound_th):
    """Boundary F-score (Perazzi et al., 2016) with a tolerance of bound_th pixels."""
    pred_dil = binary_dilation(pred, disk(bound_th))
    gt_dil = binary_dilation(gt, disk(bound_th))
    n_pred, n_gt = pred.sum(), gt.sum()
    if n_pred == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_pred > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_pred == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = (pred & gt_dil).sum() / n_pred
        recall = (gt & pred_dil).sum() / n_gt
    if precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def mean_boundary_fscore(pred, gt):
    """Average over tolerances of 1..5 pixels, as in the reference code."""
    scores = [boundary_fscore(pred, gt.astype(bool), th) for th in range(1, 6)]
    return None if scores[0] is None else float(np.mean(scores))


def surface_distance(pred, gt):
    """Symmetric mean nearest-neighbour distance between two vertex sets (mm)."""
    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    return 0.5 * (d_pg.mean() + d_gp.mean())


def vertex_error(pred, gt):
    """Mean Euclidean distance between corresponding vertices (mm)."""
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def contour_metrics(verts, affines, contours, sa_slices, sa_eval_slices, spacing_sa, spacing_la, band,
                    views=("sa", "2ch", "4ch")):
    """HD and BoundF on SAX / 2CH / 4CH for one mesh.

    contours: 'sa' (H, W, S) for the planes `sa_slices`, '2ch'/'4ch' (H, W).
    SAX HD is averaged over the slices `sa_eval_slices` (indices into sa_slices,
    slices 1/4/7 in the paper); SAX BoundF over all of them. Views not in `views`
    (no segmentation available) are reported as None.
    """
    H, W = contours["2ch"].shape
    out = {}
    hd_sa, bf_sa = [], []
    for s, z in enumerate(sa_slices):
        pts = mesh_contour_points(verts, affines["sa_inv"], z, band, H, W)
        gt = contours["sa"][..., s]
        if s in sa_eval_slices:
            hd_sa.append(contour_distances(pts, np.argwhere(gt), spacing_sa)[1])
        bf_sa.append(mean_boundary_fscore(points_to_mask(pts, H, W), gt))
    out["hd_sa"], out["bf_sa"] = _mean(hd_sa), _mean(bf_sa)
    for v in ("2ch", "4ch"):
        if v not in views:
            out[f"hd_{v}"] = out[f"bf_{v}"] = None
            continue
        pts = mesh_contour_points(verts, affines[f"{v}_inv"], 0, band, H, W)
        out[f"hd_{v}"] = contour_distances(pts, np.argwhere(contours[v]), spacing_la)[1]
        out[f"bf_{v}"] = mean_boundary_fscore(points_to_mask(pts, H, W), contours[v])
    return out


def summarize(rows):
    """mean and std of every numeric column of a list of dicts."""
    keys = [k for k in rows[0] if k != "name"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if r[k] is not None], float)
        summary[k] = dict(mean=float(vals.mean()), std=float(vals.std())) if len(vals) else None
    return summary


COLUMNS = ["hd_sa", "hd_2ch", "hd_4ch", "bf_sa", "bf_2ch", "bf_4ch", "surface_dist", "vertex_err"]


def print_table(title, summary):
    print(f"\n{title}")
    print("  " + " ".join(f"{c:>14s}" for c in COLUMNS))
    cells = []
    for c in COLUMNS:
        s = summary.get(c)
        scale = 100 if c.startswith("bf") else 1
        cells.append(f"{s['mean'] * scale:6.2f}({s['std'] * scale:5.2f})" if s else f"{'-':>14s}")
    print("  " + " ".join(f"{x:>14s}" for x in cells))
