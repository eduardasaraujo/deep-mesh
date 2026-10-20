"""Differentiable mesh-to-image rasterizer (Sec. III-C, Fig. 5).

A 3D mesh is "sliced" by a 2D view plane: every vertex is mapped into the voxel
space of that view, and its distance d to the plane gives the probability of the
vertex lying on the plane, p = exp(-tau * d^2) (Eq. 9). The in-plane coordinates
plus these probabilities form a soft 2D contour that is compared with the
ground-truth contour by the weighted Hausdorff distance (Eq. 10).
"""
import math

import torch

from deepmesh.geometry import apply_affine


def to_plane(verts, affine_inv, height, width):
    """World vertices (B, N, 3) -> in-plane (i, j) coordinates (B, N, 2) and slice coordinate k (B, N)."""
    vox = apply_affine(affine_inv, verts)
    i = vox[..., 0].clamp(0, height - 1)
    j = vox[..., 1].clamp(0, width - 1)
    return torch.stack([i, j], -1), vox[..., 2]


def plane_probability(k, z, tau):
    """Eq. (9): p = exp(-tau * |k - z|^2)."""
    return torch.exp(-tau * (k - z) ** 2)


def selection_threshold(tau):
    """Vertices with d < 1 voxel (p > exp(-tau)) form the intersection with the plane."""
    return math.exp(-tau)


def soft_contour_map(points, prob, height, width, tau):
    """Probability map (H, W) for one sample, as in Fig. 5 (right). Visualisation only."""
    keep = prob > selection_threshold(tau)
    idx = points[keep].round().long()
    flat = idx[:, 0] * width + idx[:, 1]
    out = torch.zeros(height * width, dtype=prob.dtype, device=prob.device)
    out.scatter_reduce_(0, flat, prob[keep], reduce="amax")
    return out.view(height, width)
