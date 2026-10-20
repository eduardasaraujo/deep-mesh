import math

import numpy as np
import torch

from deepmesh import losses as L
from deepmesh import mesh as M
from deepmesh.rasterizer import plane_probability, soft_contour_map, to_plane


def test_laplacian_matches_bruteforce():
    verts, faces = M.template_mesh(12, 5)
    edges = M.unique_edges(faces)
    v = torch.tensor(verts, dtype=torch.float64)[None] + 0.3 * torch.randn(1, len(verts), 3, dtype=torch.float64)
    nbrs = [set() for _ in range(len(verts))]
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    ref = np.mean([np.linalg.norm(v[0, list(n)].mean(0) - v[0, i]) for i, n in enumerate(nbrs)])
    assert math.isclose(L.laplacian_smoothing(v, torch.tensor(edges)).item(), ref, rel_tol=1e-9)


def test_chamfer_matches_bruteforce_value_and_gradient():
    x = torch.randn(2, 300, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 200, 3, dtype=torch.float64)
    d = torch.cdist(x, y) ** 2
    ref = (d.min(2)[0].mean(1) + d.min(1)[0].mean(1)).mean()
    (g_ref,) = torch.autograd.grad(ref, x)
    out = L.chamfer_distance(x, y)
    (g,) = torch.autograd.grad(out, x)
    assert torch.allclose(out, ref) and torch.allclose(g, g_ref)


def test_huber_of_constant_field():
    assert math.isclose(L.huber_regularization(torch.ones(2, 3, 4, 5, 6)).item(), 0.1, rel_tol=1e-6)


def test_rasterizer_probability():
    verts = torch.tensor([[[3.0, 4.0, 2.0], [3.0, 4.0, 2.5], [3.0, 4.0, 5.0]]])
    pts, k = to_plane(verts, torch.eye(4), 10, 10)
    p = plane_probability(k, 2, tau=3.0)
    assert torch.allclose(p[0], torch.tensor([1.0, math.exp(-0.75), math.exp(-27.0)]))
    m = soft_contour_map(pts[0], p[0], 10, 10, tau=3.0)
    assert m[3, 4] == 1.0 and m.sum() == 1.0


def test_shape_loss_pulls_points_to_the_contour():
    H = W = 32
    gt = torch.zeros(1, H, W)
    rr, cc = np.nonzero(np.abs(np.hypot(*np.mgrid[:H, :W] - 15.5) - 8) < 0.7)
    gt[0, rr, cc] = 1
    ang = torch.linspace(0, 2 * math.pi, 80)
    pts = torch.stack([15.5 + 11 * torch.cos(ang), 15.5 + 11 * torch.sin(ang)], -1)[None].requires_grad_(True)
    k = torch.zeros(1, 80)
    opt = torch.optim.SGD([pts], lr=10.0)
    first = None
    for _ in range(100):
        loss = L.plane_shape_loss(pts, k, 0, gt, tau=3.0)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    radius = (pts.detach()[0] - 15.5).norm(dim=-1).mean()
    assert loss.item() < first / 5 and abs(radius - 8) < 0.5


def test_shape_loss_empty_cases():
    pts = torch.zeros(1, 5, 2)
    far = torch.full((1, 5), 50.0)
    empty = torch.zeros(1, 8, 8)
    assert L.plane_shape_loss(pts, far, 0, empty, 3.0) == 0
    full = empty.clone()
    full[0, 2, 2] = 1
    assert math.isclose(L.plane_shape_loss(pts, far, 0, full, 3.0).item(), math.sqrt(128), rel_tol=1e-6)
