"""DICOM cines -> NIfTI, for the Kaggle Second Annual Data Science Bowl layout.

    <split>/<id>/study/sax_*/*.dcm, 2ch_*/*.dcm, 4ch_*/*.dcm

Writes sa.nii.gz (X, Y, Z, T), la_2ch.nii.gz and la_4ch.nii.gz (X, Y, 1, T) with the
convention of Bai et al.'s UK Biobank pipeline (data/biobank_utils.py): voxel i runs
along the image columns, j along the rows, and world coordinates are RAS (DICOM LPS
with x and y negated). The networks of deepmesh/real/segment.py expect this layout.

The DSB studies are irregular (repeated slices, folders holding several slices,
series with different frame counts); the choices made here are documented inline.
"""
import os
import shutil
import zipfile
from collections import Counter, defaultdict

import nibabel as nib
import numpy as np
import pydicom

LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])


def read_headers(folder):
    out = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".dcm"):
            path = os.path.join(folder, f)
            out.append((path, pydicom.dcmread(path, stop_before_pixels=True)))
    return out


def geometry(ds):
    """Unit row/column directions (RAS), pixel spacing along them, upper-left position (RAS)."""
    iop = np.array([float(x) for x in ds.ImageOrientationPatient])
    axis_x, axis_y = iop[:3] * LPS_TO_RAS, iop[3:] * LPS_TO_RAS
    dx, dy = float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0])
    pos = np.array([float(x) for x in ds.ImagePositionPatient]) * LPS_TO_RAS
    return axis_x, axis_y, dx, dy, pos


def frames_of(items):
    """Sort the images of one slice by acquisition order."""
    return sorted(items, key=lambda it: (int(getattr(it[1], "InstanceNumber", 0)), it[0]))


def load_pixels(items):
    return np.stack([pydicom.dcmread(p).pixel_array.T for p, _ in items], -1).astype(np.int16)


def slice_groups(headers):
    """Group the images of a series folder by slice position (some folders hold several slices)."""
    groups = defaultdict(list)
    for path, ds in headers:
        key = (int(getattr(ds, "SeriesNumber", 0)), tuple(np.round([float(x) for x in ds.ImagePositionPatient], 1)))
        groups[key].append((path, ds))
    return [frames_of(g) for g in groups.values()]


def stack_sax(folders):
    """Build the SAX volume from all sax_* folders; returns (volume, affine) or raises ValueError."""
    groups = [g for f in folders for g in slice_groups(read_headers(f))]
    if not groups:
        raise ValueError("no SAX images")
    # keep the most common geometry: orientation, image size, pixel spacing and frame count
    def signature(g):
        ds = g[0][1]
        return (tuple(np.round([float(x) for x in ds.ImageOrientationPatient], 2)), int(ds.Rows), int(ds.Columns),
                tuple(np.round([float(x) for x in ds.PixelSpacing], 3)), len(g))
    common = Counter(signature(g) for g in groups).most_common(1)[0][0]
    groups = [g for g in groups if signature(g) == common]

    axis_x, axis_y, dx, dy, _ = geometry(groups[0][0][1])
    normal = np.cross(axis_x, axis_y)
    # repeated acquisitions at the same position: keep the latest series
    by_pos = {}
    for g in groups:
        ds = g[0][1]
        h = round(float(np.dot(geometry(ds)[4], normal)), 0)
        if h not in by_pos or int(ds.SeriesNumber) > int(by_pos[h][0][1].SeriesNumber):
            by_pos[h] = g
    heights = np.array(sorted(by_pos))
    if len(heights) < 2:
        raise ValueError("fewer than 2 SAX slices")

    # largest run of equally spaced slices (a uniform affine needs regular spacing)
    gaps = np.diff(heights)
    gap = float(np.median(gaps))
    regular = np.abs(gaps - gap) < 0.25 * gap
    best, start = (0, 0), 0
    for i, ok in enumerate(list(regular) + [False]):
        if not ok:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = i + 1
    heights = heights[best[0]:best[1] + 1]

    slices = [by_pos[h] for h in heights]
    volume = np.stack([load_pixels(g) for g in slices], 2)  # (X, Y, Z, T)
    first, second = geometry(slices[0][0][1])[4], geometry(slices[1][0][1])[4]
    affine = np.eye(4)
    affine[:3, 0], affine[:3, 1] = axis_x * dx, axis_y * dy
    affine[:3, 2] = (second - first) / np.linalg.norm(second - first) * gap
    affine[:3, 3] = first
    return volume, affine


def single_slice(folders, n_frames):
    """One LAX cine; prefer the latest series with the same number of frames as the SAX."""
    groups = [g for f in folders for g in slice_groups(read_headers(f))]
    if not groups:
        raise ValueError("no long-axis images")
    groups.sort(key=lambda g: (len(g) == n_frames, int(g[0][1].SeriesNumber)))
    g = groups[-1]
    axis_x, axis_y, dx, dy, pos = geometry(g[0][1])
    affine = np.eye(4)
    affine[:3, 0], affine[:3, 1] = axis_x * dx, axis_y * dy
    affine[:3, 2] = np.cross(axis_x, axis_y) * float(getattr(g[0][1], "SliceThickness", 8.0))
    affine[:3, 3] = pos
    return load_pixels(g)[:, :, None], affine


def convert_study(study_dir, out_dir):
    """Convert one DSB study folder; returns a short status string."""
    series = sorted(os.listdir(study_dir))
    by_kind = {k: [os.path.join(study_dir, s) for s in series if s.startswith(k + "_")] for k in ("sax", "2ch", "4ch")}
    missing = [k for k, v in by_kind.items() if not v]
    if missing:
        return "missing " + ",".join(missing)
    sa, aff_sa = stack_sax(by_kind["sax"])
    os.makedirs(out_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(sa, aff_sa), os.path.join(out_dir, "sa.nii.gz"))
    for kind, name in (("2ch", "la_2ch"), ("4ch", "la_4ch")):
        img, aff = single_slice(by_kind[kind], sa.shape[-1])
        nib.save(nib.Nifti1Image(img, aff), os.path.join(out_dir, f"{name}.nii.gz"))
    return f"ok {sa.shape}"


def find_studies(dsb_root):
    """Map subject name -> study folder for every <id>/study under dsb_root."""
    out = {}
    for root, dirs, _ in os.walk(dsb_root):
        if os.path.basename(root) == "study":
            out[f"dsb_{os.path.basename(os.path.dirname(root))}"] = root
            dirs[:] = []
    return out


def is_converted(out_dir):
    return all(os.path.exists(os.path.join(out_dir, f"{v}.nii.gz")) for v in ("sa", "la_2ch", "la_4ch"))


def convert_zip(zip_path, native_root, work_dir, log=print):
    """Convert every study inside a Kaggle zip without extracting the whole archive.

    Studies are extracted one at a time into work_dir; zips nested in the archive
    (Kaggle bundles train.zip, validate.zip, ...) are handled the same way.
    Returns a list of (subject name, status).
    """
    os.makedirs(work_dir, exist_ok=True)
    results = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for inner in (n for n in names if n.lower().endswith(".zip")):
            tmp = os.path.join(work_dir, os.path.basename(inner))
            with zf.open(inner) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
            results += convert_zip(tmp, native_root, work_dir, log)
            os.remove(tmp)

        studies = defaultdict(list)
        for n in names:
            parts = n.split("/")
            if n.lower().endswith(".dcm") and "study" in parts[1:-1]:
                i = parts.index("study")
                studies["/".join(parts[:i + 1])].append(n)
        for prefix, members in sorted(studies.items()):
            name = f"dsb_{prefix.split('/')[-2]}"
            out_dir = os.path.join(native_root, name)
            if is_converted(out_dir):
                results.append((name, "ok (done before)"))
                continue
            tmp = os.path.join(work_dir, "study")
            shutil.rmtree(tmp, ignore_errors=True)
            for m in members:
                zf.extract(m, tmp)
            try:
                status = convert_study(os.path.join(tmp, prefix), out_dir)
            except Exception as e:  # irregular study: report and continue
                status = f"error: {e}"
            shutil.rmtree(tmp, ignore_errors=True)
            results.append((name, status))
            log(f"{name}: {status}")
    return results
