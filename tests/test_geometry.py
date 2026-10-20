import numpy as np
import torch

from deepmesh import geometry as G


def random_affine(seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A = np.eye(4)
    A[:3, :3] = q @ np.diag([2.5, 2.5, 2.5])
    A[:3, 3] = rng.normal(size=3) * 10
    return torch.tensor(A, dtype=torch.float64)


def test_affine_roundtrip():
    A = random_affine()
    p = torch.randn(2, 50, 3, dtype=torch.float64) * 20
    back = G.apply_affine(torch.linalg.inv(A), G.apply_affine(A, p))
    assert torch.allclose(back, p, atol=1e-9)


def test_voxel_grid_roundtrip_and_corners():
    shape = (64, 32, 16)
    vox = torch.rand(1, 100, 3) * torch.tensor([63.0, 31.0, 15.0])
    assert torch.allclose(G.grid_to_voxel(G.voxel_to_grid(vox, shape), shape), vox, atol=1e-4)
    corner = G.voxel_to_grid(torch.tensor([[[63.0, 0.0, 15.0]]]), shape)
    assert torch.allclose(corner, torch.tensor([[[-1.0, 1.0, 1.0]]]))


def test_sample_field_reads_the_right_voxel():
    D, H, W = 6, 7, 8
    field = torch.zeros(1, 3, D, H, W)
    field[0, :, 2, 3, 5] = torch.tensor([1.0, 2.0, 3.0])
    grid = G.voxel_to_grid(torch.tensor([[[3.0, 5.0, 2.0]]]), (H, W, D))  # (i, j, k)
    assert torch.allclose(G.sample_field(field, grid), torch.tensor([[[1.0, 2.0, 3.0]]]), atol=1e-6)


def test_warp_and_vertex_displacement_agree():
    """A blob translated by Phi must be moved back by warp() and the vertex on the
    blob centre must be moved onto the translated centre by displace_vertices()."""
    D, H, W = 16, 20, 24
    shape = (H, W, D)
    k, i, j = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij")

    def blob(ci, cj, ck):
        return torch.exp(-((i - ci) ** 2 + (j - cj) ** 2 + (k - ck) ** 2) / 8.0).float()[None, None]

    shift_vox = torch.tensor([2.0, 3.0, 1.0])  # (di, dj, dk) from frame 0 to frame t
    img0, imgt = blob(9, 11, 7), blob(9 + 2, 11 + 3, 7 + 1)
    # normalised (x, y, z) displacement equivalent to the voxel shift
    field = torch.zeros(1, 3, D, H, W)
    field[:, 0] = 2 * shift_vox[1] / (W - 1)
    field[:, 1] = 2 * shift_vox[0] / (H - 1)
    field[:, 2] = 2 * shift_vox[2] / (D - 1)

    warped = G.warp(imgt, field)
    assert (warped - img0).abs()[..., 3:-3, 3:-3, 3:-3].max() < 1e-4

    A = torch.eye(4)
    v0 = torch.tensor([[[9.0, 11.0, 7.0]]])
    vt = G.displace_vertices(v0, field, A, A, shape)
    assert torch.allclose(vt, v0 + shift_vox, atol=1e-4)
