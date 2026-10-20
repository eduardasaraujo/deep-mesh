import numpy as np

from deepmesh import mesh as M


def signed_volume(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def test_topology_is_closed_genus_zero():
    topo = M.build_topology(24, 10)
    edges = np.concatenate([topo.faces[:, [0, 1]], topo.faces[:, [1, 2]], topo.faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, 1), axis=0, return_counts=True)
    assert (counts == 2).all()  # every edge shared by exactly two faces
    V, E, F = topo.n_verts, len(counts), len(topo.faces)
    assert V - E + F == 2


def test_normals_point_out_of_the_myocardium():
    verts, faces = M.template_mesh(48, 20)
    # positive signed volume <=> outward orientation; it equals the wall volume
    s = M.TEMPLATE_SHAPE
    half_ellipsoid = lambda a, b, c: 2 / 3 * np.pi * a * b * c
    expected = half_ellipsoid(s["a_epi"], s["b_epi"], s["c_epi"]) - half_ellipsoid(s["a_endo"], s["b_endo"], s["c_endo"])
    vol = signed_volume(verts, faces)
    assert vol > 0
    assert abs(vol - expected) / expected < 0.02


def test_twist_keeps_vertices_on_the_ellipsoid():
    topo = M.build_topology(32, 12)
    shape = dict(M.TEMPLATE_SHAPE, a_epi=31.0, b_epi=27.0, twist=0.3)
    v = M.lv_vertices(shape, topo)[topo.surface == M.EPI]
    r = (v[:, 0] / 31.0) ** 2 + (v[:, 1] / 27.0) ** 2 + ((v[:, 2] - shape["z_base"]) / shape["c_epi"]) ** 2
    assert np.allclose(r, 1.0)


def test_align_template_recovers_pose():
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    x_hat, y_hat, z_hat = q.T
    centre = np.array([5.0, -3.0, 12.0])
    D = 20

    def affine(e1, e2, n, origin):
        A = np.eye(4)
        A[:3, 0], A[:3, 1], A[:3, 2], A[:3, 3] = e1, e2, n, origin
        return A

    # SAX: k goes from base to apex (-z); middle slice through the centre
    sa = affine(x_hat * 2, y_hat * 2, -z_hat * 2, centre - (-z_hat * 2) * (D - 1) / 2 - 7 * x_hat)
    a2 = affine(z_hat, y_hat, x_hat, centre - 30 * z_hat)   # 2CH plane: span(y, z)
    a4 = affine(z_hat, x_hat, y_hat, centre + 9 * x_hat)    # 4CH plane: span(x, z)
    tpl, _ = M.template_mesh(16, 6)
    aligned = M.align_template(tpl, sa, a2, a4, D)
    assert np.allclose(aligned, tpl @ q.T + centre, atol=1e-6)
