"""Convert native cine NIfTIs + segmentations into the DeepMesh subject format.

Input (one folder per subject, the layout of Bai et al.'s UK Biobank pipeline):
    sa.nii.gz, la_2ch.nii.gz, la_4ch.nii.gz       cines (X, Y, Z, T)
    seg_sa.nii.gz, seg_la_4ch.nii.gz              labels of deepmesh/real/segment.py

Output: the files read by deepmesh/data.py (see scripts/generate_synthetic.py), with
* the SAX stack resampled so that `data.sa_slices` coincide exactly with acquired
  slices (the paper's 128x128x64 grid with slices 12, 17, ..., 52); intensities and
  myocardium labels between acquired slices are interpolated (labels via signed
  distances, i.e. shape-based interpolation);
* 2CH/4CH resampled in their own plane and centred on the LV;
* GT surfaces from marching cubes on the interpolated SAX myocardium (ED and ES);
* the template aligned from the image geometry only (Fig. 11);
* quality-control measures, so that failed segmentations can be discarded.
"""
import json
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, map_coordinates
from skimage.measure import marching_cubes

from deepmesh import mesh as M
from deepmesh.real.segment import LV_LABEL, MYO_LABEL
from deepmesh.synthetic.phantom import make_affine, voxel_grid

OUTSIDE = 1e3  # signed distance used where no label exists
DEVICE = {"device": torch.device("cpu")}  # set by scripts/prepare_real.py (GPU when available)

# a thin myocardium cut through ~10 mm SAX slices gives a low Dice even when aligned:
# the 4CH/SAX threshold is permissive and the distance in mm is also recorded in qc.csv
QC_LIMITS = dict(min_myo_slices=6, max_plane_dist_mm=15.0, min_lax_myo_px=20, min_dice_4ch_sax=0.3,
                 ef_range=(0.1, 0.9))


def unit(v):
    return v / np.linalg.norm(v)


def to_voxel(affine, world):
    inv = np.linalg.inv(affine)
    return world @ inv[:3, :3].T + inv[:3, 3]


def sample(volume, affine, world, order=1, cval=0.0):
    """Interpolate a 3D volume (voxel->world `affine`) at world points (..., 3)."""
    vox = to_voxel(affine, world.reshape(-1, 3)).T
    return map_coordinates(volume, vox, order=order, mode="constant", cval=cval).reshape(world.shape[:-1])


def sample_frames(volume, affine, world, cval=0.0):
    """Trilinear interpolation of every frame of a (X, Y, Z, T) volume at world points (..., 3).

    One torch grid_sample call for all frames (multi-threaded), instead of one
    map_coordinates call per frame. Returns (..., T) float32.
    """
    X, Y, Z, T = volume.shape
    dev = DEVICE["device"]
    vox = torch.from_numpy(to_voxel(affine, world.reshape(-1, 3))).float().to(dev)
    size = torch.tensor([X, Y, Z], dtype=torch.float32, device=dev)
    grid = (2 * vox / (size - 1).clamp(min=1) - 1).view(1, 1, 1, -1, 3)  # (x, y, z) = (i, j, k)
    vol = torch.from_numpy(np.ascontiguousarray(volume.transpose(3, 2, 1, 0), dtype=np.float32))
    vol = vol.to(dev)[None] - cval
    out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return (out[0, :, 0, 0].T.cpu().numpy() + cval).reshape(world.shape[:-1] + (T,))


def sample_plane(image, affine, world, order=1):
    """Interpolate a single-slice image (X, Y) at world points lying on its plane."""
    vox = to_voxel(affine, world.reshape(-1, 3))[:, :2].T
    return map_coordinates(image, vox, order=order, mode="constant", cval=0.0).reshape(world.shape[:-1])


def signed_distance_slices(mask):
    """Slice-wise 2D signed distance (pixels, negative inside) of a (X, Y, Z) mask."""
    out = np.full(mask.shape, OUTSIDE, np.float32)
    for z in range(mask.shape[2]):
        m = mask[..., z]
        if m.any():
            out[..., z] = distance_transform_edt(~m) - distance_transform_edt(m)
    return out


def to_uint8(image, lo, hi):
    return (np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def surface(sdf, affine, sigma=(1.0, 1.0, 1.5)):
    """Closed myocardium surface from a (H, W, D) signed distance volume (world vertices + faces).

    Only the largest connected component is kept (removes stray blobs of the
    segmentation) and the mask is smoothed a little, mostly across slices, so that
    the staircase left by the ~10 mm slice spacing does not dominate the GT mesh.
    """
    mask = sdf < 0
    comps, n = label(mask)
    if n > 1:
        mask = comps == (np.argmax(np.bincount(comps.ravel())[1:]) + 1)
    padded = np.pad(gaussian_filter(mask.astype(np.float32), sigma), 1)
    verts, faces, _, _ = marching_cubes(padded, level=0.5)
    world = (verts - 1) @ affine[:3, :3].T + affine[:3, 3]
    return world.astype(np.float32), faces.astype(np.int64)


def load(path):
    nim = nib.load(path)
    return np.asanyarray(nim.dataobj), nim.affine.astype(np.float64)


def prepare_subject(native_dir, d, template_local):
    """Return (outputs dict, qc dict). outputs is None when the subject cannot be used."""
    img_sa, A_sa = load(os.path.join(native_dir, "sa.nii.gz"))
    seg_sa, _ = load(os.path.join(native_dir, "seg_sa.nii.gz"))
    img_sa = img_sa.astype(np.float32)
    X, Y, S, T = img_sa.shape
    H, W, D = d.height, d.width, d.depth
    lv, myo = seg_sa == LV_LABEL["sa"], seg_sa == MYO_LABEL["sa"]
    qc = dict(n_slices=S, n_frames=T)

    ed = 0
    lv_vol = lv.sum((0, 1, 2))
    es = int(np.argmin(lv_vol))
    qc.update(es_frame=es, ef=float(1 - lv_vol[es] / max(lv_vol[ed], 1)))

    # base -> apex order of the native slices: the LV cross-section shrinks towards the apex
    area = (lv | myo)[..., ed].sum((0, 1)).astype(float)
    with_myo = np.nonzero(myo[..., ed].sum((0, 1)) > 0)[0]
    qc["myo_slices"] = int(len(with_myo))
    n_acq = len(d.sa_slices)
    stride = d.sa_slices[1] - d.sa_slices[0]
    assert np.all(np.diff(d.sa_slices) == stride), "data.sa_slices must be equally spaced"
    if S < n_acq:
        qc["reject"] = f"SAX stack has fewer than {n_acq} slices"
        return None, qc
    if len(with_myo) < 3:
        qc["reject"] = "few SAX slices with myocardium"
        return None, qc
    order = np.arange(S) if np.polyfit(with_myo, area[with_myo], 1)[0] < 0 else np.arange(S)[::-1]

    # n_acq consecutive slices that contain the most myocardium
    myo_area = myo[..., ed].sum((0, 1))[order]
    w0 = int(np.argmax([myo_area[w:w + n_acq].sum() for w in range(S - n_acq + 1)]))
    sel = order[w0:w0 + n_acq]

    # target SAX grid: native in-plane axes, k from base to apex, acquired slices on d.sa_slices
    gap = np.linalg.norm(A_sa[:3, 2])
    n_t = unit(A_sa[:3, 2]) * (1 if order[0] == 0 else -1)
    e1, e2 = unit(A_sa[:3, 0]), unit(A_sa[:3, 1])
    sp_i, sp_j, sp_k = d.sa_spacing[0], d.sa_spacing[1], gap / stride
    idx = np.argwhere((lv | myo)[..., ed][:, :, sel])
    idx[:, 2] = sel[idx[:, 2]]
    centre = (idx @ A_sa[:3, :3].T + A_sa[:3, 3]).mean(0)
    p0 = A_sa[:3, :3] @ np.array([0.0, 0.0, sel[0]]) + A_sa[:3, 3]
    c_plane = centre - np.dot(centre - p0, n_t) * n_t
    A_t = np.eye(4)
    A_t[:3, 0], A_t[:3, 1], A_t[:3, 2] = e1 * sp_i, e2 * sp_j, n_t * sp_k
    A_t[:3, 3] = c_plane - (H - 1) / 2 * sp_i * e1 - (W - 1) / 2 * sp_j * e2 - d.sa_slices[0] * sp_k * n_t

    grid_sa = voxel_grid(A_t, (H, W, D))
    lo, hi = np.percentile(img_sa[::2, ::2], (1, 99))
    sdf = np.stack([signed_distance_slices(myo[..., t]) for t in range(T)], -1)
    sa = to_uint8(sample_frames(img_sa, A_sa, grid_sa), lo, hi)
    sdf_t = sample_frames(sdf, A_sa, grid_sa, cval=OUTSIDE)  # (H, W, D, T) on the target grid
    sa_myo = (sdf_t < 0).astype(np.uint8)

    # long-axis views, centred on the LV and resampled in their own plane
    out = dict(images={"sa": sa}, labels={"sa": sa_myo}, affines={"sa": A_t})
    for view, name in (("2ch", "la_2ch"), ("4ch", "la_4ch")):
        img, A_l = load(os.path.join(native_dir, f"{name}.nii.gz"))
        img = img.astype(np.float32)
        nl = unit(np.cross(A_l[:3, 0], A_l[:3, 1]))
        dist = float(np.dot(centre - A_l[:3, 3], nl))
        qc[f"dist_{view}_mm"] = abs(dist)
        axes = np.stack([unit(A_l[:3, 0]), unit(A_l[:3, 1]), nl], 1)
        A_lt = make_affine(axes, d.lax_spacing, centre - dist * nl, (H, W, 1))
        grid = voxel_grid(A_lt, (H, W, 1))
        frames = np.round(np.arange(T) * img.shape[-1] / T).astype(int) % img.shape[-1]
        lo_l, hi_l = np.percentile(img[::2, ::2], (1, 99))
        out["images"][view] = np.stack([to_uint8(sample_plane(img[..., 0, f], A_l, grid), lo_l, hi_l)
                                        for f in frames], -1)
        out["affines"][view] = A_lt
        seg_path = os.path.join(native_dir, f"seg_{name}.nii.gz")
        if os.path.exists(seg_path) and name in MYO_LABEL:
            seg, _ = load(seg_path)
            lab = (seg == MYO_LABEL[name]).astype(np.float32)
            out["labels"][view] = np.stack([sample_plane(lab[..., 0, f], A_l, grid, order=0)
                                            for f in frames], -1).astype(np.uint8)

    # QC: the 4CH myocardium must agree with the SAX myocardium cut by the 4CH plane
    if "4ch" in out["labels"]:
        lab4 = out["labels"]["4ch"][..., 0, :]
        qc["myo_4ch_px_min"] = int(min(lab4[..., ed].sum(), lab4[..., es].sum()))
        grid4 = voxel_grid(out["affines"]["4ch"], (H, W, 1))[..., 0, :]
        vox = to_voxel(A_sa, grid4.reshape(-1, 3)).reshape(H, W, 3)
        valid = (vox[..., 2] >= 0) & (vox[..., 2] <= S - 1)
        sax_cut = sample(sdf[..., ed], A_sa, grid4, cval=OUTSIDE) < 0
        a, b = lab4[..., ed].astype(bool) & valid, sax_cut & valid
        qc["dice_4ch_sax"] = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
        if a.any() and b.any():  # mean symmetric distance between the two myocardia (mm)
            d_ab = distance_transform_edt(~b)[a].mean()
            d_ba = distance_transform_edt(~a)[b].mean()
            qc["dist_4ch_sax_mm"] = float(0.5 * (d_ab + d_ba) * d.lax_spacing[0])

    # real LAX planes rarely cross exactly at the LV centre: anchor the axis on the segmented centroid
    out["template_aligned"] = M.align_template(template_local, A_t, out["affines"]["2ch"],
                                               out["affines"]["4ch"], D, centre=centre).astype(np.float32)
    for key, t in (("gt_surface_ed", ed), ("gt_surface_es", es)):
        out[key] = surface(sdf_t[..., t], A_t)
    out["meta"] = dict(ed_frame=ed, es_frame=es, n_frames=T, source=os.path.abspath(native_dir))
    qc["reject"] = quality_check(qc)
    return out, qc


def quality_check(qc, limits=QC_LIMITS):
    """Reason for rejection, or '' when the subject passes."""
    if qc["myo_slices"] < limits["min_myo_slices"]:
        return "few SAX slices with myocardium"
    if not limits["ef_range"][0] < qc["ef"] < limits["ef_range"][1]:
        return "implausible ejection fraction"
    for view in ("2ch", "4ch"):
        if qc[f"dist_{view}_mm"] > limits["max_plane_dist_mm"]:
            return f"{view} plane far from the LV"
    if qc.get("myo_4ch_px_min", 0) < limits["min_lax_myo_px"]:
        return "no 4CH myocardium"
    if qc.get("dice_4ch_sax", 0) < limits["min_dice_4ch_sax"]:
        return "4CH and SAX disagree (breath-hold or segmentation)"
    return ""


def write_subject(path, out):
    os.makedirs(path, exist_ok=True)
    for view in ("sa", "2ch", "4ch"):
        nib.save(nib.Nifti1Image(out["images"][view], out["affines"][view]), os.path.join(path, f"{view}.nii.gz"))
        if view in out["labels"]:
            nib.save(nib.Nifti1Image(out["labels"][view], out["affines"][view]),
                     os.path.join(path, f"{view}_myo.nii.gz"))
    np.save(os.path.join(path, "template_aligned.npy"), out["template_aligned"])
    for key in ("gt_surface_ed", "gt_surface_es"):
        verts, faces = out[key]
        np.savez(os.path.join(path, f"{key}.npz"), verts=verts, faces=faces)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(out["meta"], f, indent=1)
