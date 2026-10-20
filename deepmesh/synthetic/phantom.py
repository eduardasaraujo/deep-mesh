"""Synthetic beating-LV phantom with SAX, 2CH and 4CH cine images.

UK Biobank data cannot be redistributed, so this module generates subjects with
the same structure as the paper's data: a resampled SAX stack (H, W, D, T), single
slice 2CH/4CH cines (H, W, 1, T), myocardium labels, NIfTI affines that relate all
views in world space, and 3D ground-truth meshes.

Because the motion is analytic, the generator also provides what real data never
has: ground-truth vertex correspondences over time (used as an extra metric).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from deepmesh import mesh as M

BLOOD, ATRIUM, MYO = 0.85, 0.72, 0.28


def _rotation_towards(rng, max_deg):
    """Random rotation matrix with angle <= max_deg."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(0, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def sample_subject(rng, n_frames):
    a_epi = rng.uniform(26, 33)
    b_epi = a_epi * rng.uniform(0.9, 1.05)
    thick = rng.uniform(6, 9)
    c_epi = rng.uniform(76, 92)
    ed = dict(a_endo=a_epi - thick, b_endo=b_epi - thick, c_endo=c_epi - rng.uniform(5, 8),
              a_epi=a_epi, b_epi=b_epi, c_epi=c_epi, z_base=c_epi / 2, twist=0.0)
    motion = dict(endo_radial=rng.uniform(0.22, 0.34), epi_radial=rng.uniform(0.06, 0.11),
                  long_short=rng.uniform(0.10, 0.16), apex_up=rng.uniform(0.02, 0.06),
                  twist=np.deg2rad(rng.uniform(8, 18)), rv=rng.uniform(0.15, 0.3))

    # pose of the LV frame in world space: oblique long axis, random roll
    rot = _rotation_towards(rng, 180)
    centre = rng.normal(0, 5, 3)

    rv = dict(gap=rng.uniform(1.5, 3.5), ax=rng.uniform(14, 20), ay_scale=rng.uniform(1.05, 1.25),
              az_scale=rng.uniform(0.65, 0.8))
    k = rng.normal(size=(6, 3)) / rng.uniform(15, 50, size=(6, 1))
    background = dict(freq=k, phase=rng.uniform(0, 2 * np.pi, 6), amp=rng.uniform(0.02, 0.06, 6),
                      level=rng.uniform(0.18, 0.3))
    es_frame = int(round(rng.uniform(0.32, 0.40) * n_frames))
    return dict(ed=ed, motion=motion, rot=rot, centre=centre, rv=rv, background=background,
                es_frame=es_frame, n_frames=n_frames)


def contraction(t, n_frames, es_frame):
    """Phase f in [0, 1]: 0 at ED (t=0), 1 at ES, smooth systole and diastole."""
    x, e = t / n_frames, es_frame / n_frames
    if x <= e:
        return 0.5 * (1 - np.cos(np.pi * x / e))
    return 0.5 * (1 + np.cos(np.pi * (x - e) / (1 - e)))


def shape_at(ed, motion, f):
    """LV shape parameters at contraction phase f (apex of the epicardium is fixed)."""
    descent = motion["long_short"] * f * ed["c_epi"]
    return dict(a_endo=ed["a_endo"] * (1 - motion["endo_radial"] * f),
                b_endo=ed["b_endo"] * (1 - motion["endo_radial"] * f),
                c_endo=(ed["c_endo"] - descent) * (1 - motion["apex_up"] * f),
                a_epi=ed["a_epi"] * (1 - motion["epi_radial"] * f),
                b_epi=ed["b_epi"] * (1 - motion["epi_radial"] * f),
                c_epi=ed["c_epi"] - descent,
                z_base=ed["z_base"] - descent,
                twist=motion["twist"] * f)


def _inside(p, a, b, c, z0):
    return (p[..., 0] / a) ** 2 + (p[..., 1] / b) ** 2 + ((p[..., 2] - z0) / c) ** 2 <= 1


def render(world, subj, f):
    """Intensity in [0, 1] and LV-myocardium label at world points (..., 3)."""
    s = shape_at(subj["ed"], subj["motion"], f)
    p = (world - subj["centre"]) @ subj["rot"]  # world -> local LV frame
    below = p[..., 2] <= s["z_base"]
    epi = _inside(p, s["a_epi"], s["b_epi"], s["c_epi"], s["z_base"]) & below
    endo = _inside(p, s["a_endo"], s["b_endo"], s["c_endo"], s["z_base"]) & below
    myo = epi & ~endo

    rv, m = subj["rv"], subj["motion"]
    ax = rv["ax"] * (1 - m["rv"] * f)
    rv_c = np.array([s["a_epi"] + rv["gap"] + 0.35 * ax, 0.0, s["z_base"] - 0.45 * s["c_epi"]])
    q = p - rv_c
    ay, az = s["b_epi"] * rv["ay_scale"], s["c_epi"] * rv["az_scale"]
    rv_blood = _inside(q, ax, ay, az, 0.0) & ~epi
    rv_wall = _inside(q, ax + 3, ay + 3, az + 3, 0.0) & ~rv_blood & ~epi
    atrium = ((p[..., 0] / (s["a_endo"] * 1.1)) ** 2 + (p[..., 1] / (s["b_endo"] * 1.1)) ** 2 <= 1) \
        & (p[..., 2] > s["z_base"]) & (p[..., 2] < s["z_base"] + 30)

    bg = subj["background"]
    img = bg["level"] + np.cos(world @ bg["freq"].T + bg["phase"]) @ bg["amp"]
    img = np.where(rv_wall | myo, MYO, img)
    img = np.where(atrium, ATRIUM, img)
    img = np.where(endo | rv_blood, BLOOD, img)
    return img, myo


def make_affine(axes, spacing, centre, shape):
    """voxel->world affine with columns axes*spacing and the image centre at `centre`."""
    A = np.eye(4)
    A[:3, :3] = axes * np.asarray(spacing)
    A[:3, 3] = centre - A[:3, :3] @ ((np.asarray(shape) - 1) / 2)
    return A


def view_affines(rng, subj, cfg):
    """SAX stack (k from base to apex) and the two long-axis planes."""
    R, c = subj["rot"], subj["centre"]
    x, y, z = R.T
    H, W, D = cfg.height, cfg.width, cfg.depth

    tilt = _rotation_towards(rng, 6)
    roll = rng.uniform(0, 2 * np.pi)
    e1 = np.cos(roll) * x + np.sin(roll) * y
    e2 = np.cross(-z, e1)
    sa_axes = tilt @ np.stack([e1, e2, -z], 1)
    sa = make_affine(sa_axes, cfg.sa_spacing, c + rng.normal(0, 4, 3), (H, W, D))

    def lax(e1, e2, n):
        axes = _rotation_towards(rng, 5) @ np.stack([e1, e2, n], 1)
        centre = c + rng.normal(0, 4) * axes[:, 0] + rng.normal(0, 4) * axes[:, 1] + rng.normal(0, 1.5) * axes[:, 2]
        return make_affine(axes, cfg.lax_spacing, centre, (H, W, 1))

    return dict(sa=sa, la2ch=lax(y, z, x), la4ch=lax(z, x, y))


def voxel_grid(affine, shape):
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).astype(np.float64)
    return idx @ affine[:3, :3].T + affine[:3, 3]


def _to_uint8(img, rng, sigma):
    img = gaussian_filter(img, sigma) + rng.normal(0, 0.035, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def generate_subject(rng, cfg, topo, gt_topo, template_local):
    """Return a dict with every array that is written to disk for one subject."""
    T = cfg.n_frames
    subj = sample_subject(rng, T)
    aff = view_affines(rng, subj, cfg)
    grids = dict(sa=voxel_grid(aff["sa"], (cfg.height, cfg.width, cfg.depth)),
                 la2ch=voxel_grid(aff["la2ch"], (cfg.height, cfg.width, 1)),
                 la4ch=voxel_grid(aff["la4ch"], (cfg.height, cfg.width, 1)))

    images = {v: np.zeros(g.shape[:3] + (T,), np.uint8) for v, g in grids.items()}
    labels = {v: np.zeros(g.shape[:3] + (T,), np.uint8) for v, g in grids.items()}
    verts = np.zeros((T, topo.n_verts, 3), np.float32)
    for t in range(T):
        f = contraction(t, T, subj["es_frame"])
        for v, g in grids.items():
            img, myo = render(g, subj, f)
            sigma = (0.6, 0.6, 0.6) if v == "sa" else (0.6, 0.6, 0)
            images[v][..., t] = _to_uint8(img, rng, sigma)
            labels[v][..., t] = myo
        local = M.lv_vertices(shape_at(subj["ed"], subj["motion"], f), topo)
        verts[t] = local @ subj["rot"].T + subj["centre"]

    def gt_surface(t):
        s = shape_at(subj["ed"], subj["motion"], contraction(t, T, subj["es_frame"]))
        return (M.lv_vertices(s, gt_topo) @ subj["rot"].T + subj["centre"]).astype(np.float32)

    template = M.align_template(template_local, aff["sa"], aff["la2ch"], aff["la4ch"], cfg.depth)
    return dict(subject=subj, affines=aff, images=images, labels=labels, gt_vertices=verts,
                gt_surface_ed=gt_surface(0), gt_surface_es=gt_surface(subj["es_frame"]),
                template_aligned=template.astype(np.float32))
