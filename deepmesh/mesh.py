"""Left-ventricle myocardium mesh: template topology, parametric shape and helpers.

The paper uses the LV template of Bai et al. (2015), which is not redistributable.
Here the myocardium is modelled as two truncated ellipsoids (endocardium and
epicardium) joined at the basal ring, giving a closed genus-0 surface. Every mesh
built from the same topology has the same vertices and faces, which is exactly the
property DeepMesh relies on (vertex correspondence across frames and subjects).

Local LV frame: +z points from apex to base, the base plane is z = z_base and the
apex of each surface is at z_base - c. +x points towards the right ventricle
(this axis lies in the 4CH plane and is the normal of the 2CH plane).
"""
from dataclasses import dataclass

import numpy as np

ENDO, EPI = 0, 1


@dataclass
class Topology:
    u: np.ndarray        # polar angle from the apex, (N,)
    phi: np.ndarray      # circumferential angle, (N,)
    surface: np.ndarray  # ENDO or EPI, (N,)
    faces: np.ndarray    # (F, 3) int64
    n_around: int
    n_long: int

    @property
    def n_verts(self):
        return len(self.u)


def build_topology(n_around, n_long):
    """Vertices: per surface one apex vertex followed by n_long rings of n_around."""
    per_surface = 1 + n_long * n_around
    u, phi, surf, faces = [], [], [], []
    for s in (ENDO, EPI):
        u.append(0.0)
        phi.append(0.0)
        surf.append(s)
        for k in range(1, n_long + 1):
            for m in range(n_around):
                u.append(0.5 * np.pi * k / n_long)
                phi.append(2 * np.pi * m / n_around)
                surf.append(s)

    def ring(s, k, m):
        return s * per_surface + 1 + (k - 1) * n_around + (m % n_around)

    for s in (ENDO, EPI):
        tri = []
        apex = s * per_surface
        for m in range(n_around):
            tri.append((apex, ring(s, 1, m), ring(s, 1, m + 1)))
        for k in range(1, n_long):
            for m in range(n_around):
                tri.append((ring(s, k, m), ring(s, k + 1, m), ring(s, k + 1, m + 1)))
                tri.append((ring(s, k, m), ring(s, k + 1, m + 1), ring(s, k, m + 1)))
        tri = np.array(tri)
        # this winding points into the cavity: correct for the endocardium,
        # reversed for the epicardium so that all normals leave the myocardium
        faces.append(tri if s == ENDO else tri[:, ::-1])

    base = []
    for m in range(n_around):
        endo_a, endo_b = ring(ENDO, n_long, m), ring(ENDO, n_long, m + 1)
        epi_a, epi_b = ring(EPI, n_long, m), ring(EPI, n_long, m + 1)
        base.append((endo_a, epi_a, epi_b))
        base.append((endo_a, epi_b, endo_b))
    faces.append(np.array(base))

    return Topology(np.array(u), np.array(phi), np.array(surf),
                    np.concatenate(faces).astype(np.int64), n_around, n_long)


def lv_vertices(shape, topo):
    """Vertex positions (N, 3) in the local LV frame for a shape-parameter dict.

    shape keys: a_endo, b_endo, c_endo, a_epi, b_epi, c_epi, z_base, twist (rad).
    The twist shifts the circumferential angle linearly from apex to base, which
    keeps every vertex on its ellipsoid.
    """
    epi = topo.surface == EPI
    a = np.where(epi, shape["a_epi"], shape["a_endo"])
    b = np.where(epi, shape["b_epi"], shape["b_endo"])
    c = np.where(epi, shape["c_epi"], shape["c_endo"])
    height = 1 - np.cos(topo.u)  # 0 at the apex, 1 at the base
    phi = topo.phi + shape.get("twist", 0.0) * (0.5 - height)
    x = a * np.sin(topo.u) * np.cos(phi)
    y = b * np.sin(topo.u) * np.sin(phi)
    z = shape["z_base"] - c * np.cos(topo.u)
    return np.stack([x, y, z], -1)


# Mean ED shape used as the template (mm)
TEMPLATE_SHAPE = dict(a_endo=23.0, b_endo=23.0, c_endo=76.0,
                      a_epi=30.0, b_epi=30.0, c_epi=84.0, z_base=42.0, twist=0.0)


def template_mesh(n_around, n_long):
    topo = build_topology(n_around, n_long)
    return lv_vertices(TEMPLATE_SHAPE, topo), topo.faces


def unique_edges(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def align_template(template_local, sa_affine, affine_2ch, affine_4ch, sa_depth):
    """Move the template into subject space using only image-header geometry.

    This is the pre-processing step of Fig. 11: the long axis is the intersection
    line of the 2CH and 4CH planes, and the LV centre is where that line crosses
    the middle SAX slice.
    """
    def unit(v):
        return v / np.linalg.norm(v)

    n_2ch = unit(affine_2ch[:3, 2])
    n_4ch = unit(affine_4ch[:3, 2])
    n_sa = unit(sa_affine[:3, 2])

    axis = unit(np.cross(n_2ch, n_4ch))
    if np.dot(axis, -n_sa) < 0:  # SAX slices go from base to apex
        axis = -axis
    x_hat = unit(n_2ch - np.dot(n_2ch, axis) * axis)
    y_hat = np.cross(axis, x_hat)
    rot = np.stack([x_hat, y_hat, axis], 1)

    mid = sa_affine @ np.array([0.0, 0.0, (sa_depth - 1) / 2, 1.0])
    normals = np.stack([n_2ch, n_4ch, n_sa])
    offsets = np.array([n_2ch @ affine_2ch[:3, 3], n_4ch @ affine_4ch[:3, 3], n_sa @ mid[:3]])
    centre = np.linalg.solve(normals, offsets)
    return template_local @ rot.T + centre


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for t in faces + 1:
            f.write(f"f {t[0]} {t[1]} {t[2]}\n")
