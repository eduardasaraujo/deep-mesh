"""Coordinate systems and sampling utilities.

Conventions used everywhere in this package
-------------------------------------------
* World space: millimetres (scanner coordinates), points as (..., 3).
* Voxel space of an image: (i, j, k) indices, i.e. NIfTI order. ``affine`` maps
  voxel -> world, ``affine_inv`` maps world -> voxel.
* Network tensors store a SAX volume as (B, C, D, H, W) with D=k, H=i, W=j.
* ``F.grid_sample`` expects normalised coordinates ordered (x, y, z) = (W, H, D),
  i.e. (j, i, k), in [-1, 1] with ``align_corners=True``.
* Displacement fields predicted by the networks are (B, 3, D, H, W) in that same
  normalised (x, y, z) order. The same field is used both to move mesh vertices
  and to warp images (L_sim), so the two uses are consistent.

Note: the reference code normalised vertices as (i, j, k) while warping images as
(j, i, k); this only worked because H == W. Here a single convention is used.
"""
import torch
import torch.nn.functional as F


def apply_affine(affine, points):
    """Apply a (B, 4, 4) or (4, 4) affine to points (B, N, 3) or (N, 3)."""
    rot = affine[..., :3, :3]
    trans = affine[..., :3, 3]
    return points @ rot.transpose(-1, -2) + trans.unsqueeze(-2)


def voxel_to_grid(vox, shape):
    """(i, j, k) voxel coordinates -> normalised (x, y, z) grid coordinates.

    shape = (H, W, D) of the volume.
    """
    H, W, D = shape
    i, j, k = vox.unbind(-1)
    return torch.stack([2 * j / (W - 1) - 1, 2 * i / (H - 1) - 1, 2 * k / (D - 1) - 1], -1)


def grid_to_voxel(grid, shape):
    H, W, D = shape
    x, y, z = grid.unbind(-1)
    return torch.stack([(y + 1) * (H - 1) / 2, (x + 1) * (W - 1) / 2, (z + 1) * (D - 1) / 2], -1)


def sample_field(field, grid):
    """Trilinearly sample a dense field (B, C, D, H, W) at grid points (B, N, 3).

    This is the "grid sampler" S(., .) of Eq. (1) and Eq. (6). Returns (B, N, C).
    """
    out = F.grid_sample(field, grid[:, None, None], mode="bilinear", align_corners=True)
    return out[:, :, 0, 0].transpose(1, 2)


def displace_vertices(verts, field, sa_affine, sa_affine_inv, shape):
    """Move world-space vertices (B, N, 3) with a dense field defined on the SAX grid.

    Implements V' = V + S(Phi, V) (Eq. 2 / Eq. 7), with the sampling done in the
    normalised SAX image space and the result mapped back to world space.
    """
    grid = voxel_to_grid(apply_affine(sa_affine_inv, verts), shape)
    moved = grid + sample_field(field, grid)
    return apply_affine(sa_affine, grid_to_voxel(moved, shape))


def identity_grid(batch, size, device):
    """Identity sampling grid (B, D, H, W, 3) for a volume of size (D, H, W)."""
    theta = torch.eye(3, 4, device=device).unsqueeze(0).expand(batch, 3, 4)
    return F.affine_grid(theta, (batch, 1, *size), align_corners=True)


def warp(image, field):
    """Spatial transformer: returns image(p + Phi(p)), i.e. I_t o Phi_{0->t} in Eq. (8).

    image: (B, C, D, H, W); field: (B, 3, D, H, W) normalised displacement.
    """
    grid = identity_grid(image.shape[0], image.shape[2:], image.device)
    grid = grid + field.permute(0, 2, 3, 4, 1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
