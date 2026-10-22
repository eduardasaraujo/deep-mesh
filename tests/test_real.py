import os

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from deepmesh.real.dicom import convert_study
from deepmesh.synthetic.phantom import make_affine

RAS_TO_LPS = np.array([-1.0, -1.0, 1.0])


def write_dicom(path, image, affine, series, instance, thickness=8.0):
    """Inverse of the converter's convention: pixel_array = image.T, LPS coordinates."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    dx, dy = np.linalg.norm(affine[:3, 0]), np.linalg.norm(affine[:3, 1])
    ds.ImageOrientationPatient = list(np.concatenate([affine[:3, 0] / dx, affine[:3, 1] / dy]) * np.tile(RAS_TO_LPS, 2))
    ds.ImagePositionPatient = list(affine[:3, 3] * RAS_TO_LPS)
    ds.PixelSpacing = [dy, dx]
    ds.SliceThickness = thickness
    ds.SeriesNumber, ds.InstanceNumber = series, instance
    pix = np.ascontiguousarray(image.T.astype(np.uint16))
    ds.Rows, ds.Columns = pix.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = pix.tobytes()
    ds.save_as(path, enforce_file_format=True)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("dsb") / "train" / "7" / "study"
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    X, Y, Z, T = 24, 20, 6, 5
    sa_aff = make_affine(q, (1.4, 1.4, 10.0), np.array([5.0, -3.0, 20.0]), (X, Y, Z))
    sa = rng.integers(0, 1000, size=(X, Y, Z, T))
    for z in range(Z):
        aff = sa_aff.copy()
        aff[:3, 3] = sa_aff[:3, :3] @ [0, 0, z] + sa_aff[:3, 3]
        # slices 0-1 share a folder (one series), the others one folder each
        folder = root / ("sax_5" if z < 2 else f"sax_{z + 4}")
        folder.mkdir(parents=True, exist_ok=True)
        for t in range(T):
            write_dicom(str(folder / f"IM-{z}-{t:04d}.dcm"), sa[..., z, t], aff, series=10 + max(z, 1), instance=t + 1)
    # an older repeat of slice 3 with different content: must be ignored
    (root / "sax_99").mkdir()
    aff3 = sa_aff.copy()
    aff3[:3, 3] = sa_aff[:3, :3] @ [0, 0, 3] + sa_aff[:3, 3]
    for t in range(T):
        write_dicom(str(root / "sax_99" / f"IM-old-{t:04d}.dcm"), np.zeros((X, Y)), aff3, series=1, instance=t + 1)
    lax_aff = make_affine(np.roll(q, 1, axis=1), (1.5, 1.5, 6.0), np.zeros(3), (X, Y, 1))
    lax = rng.integers(0, 1000, size=(X, Y, T))
    for view in ("2ch_20", "4ch_21"):
        (root / view).mkdir()
        for t in range(T):
            write_dicom(str(root / view / f"IM-{t:04d}.dcm"), lax[..., t], lax_aff, series=30, instance=t + 1, thickness=6.0)
    return root, sa, sa_aff, lax, lax_aff


def test_convert_study_recovers_images_and_geometry(study, tmp_path):
    root, sa, sa_aff, lax, lax_aff = study
    assert convert_study(str(root), str(tmp_path)).startswith("ok")
    nim = nib.load(tmp_path / "sa.nii.gz")
    data = np.asanyarray(nim.dataobj)
    if np.dot(nim.affine[:3, 2], sa_aff[:3, 2]) < 0:  # slice order may be reversed
        data = data[:, :, ::-1]
        assert np.allclose(nim.affine[:3, :3] @ [0, 0, data.shape[2] - 1] + nim.affine[:3, 3], sa_aff[:3, 3], atol=1e-3)
    else:
        assert np.allclose(nim.affine, sa_aff, atol=1e-3)
    assert np.array_equal(data, sa)
    la = nib.load(tmp_path / "la_4ch.nii.gz")
    assert np.array_equal(np.asanyarray(la.dataobj)[:, :, 0], lax)
    assert np.allclose(la.affine[:3, :2], lax_aff[:3, :2], atol=1e-4)
    assert np.allclose(la.affine[:3, 3], lax_aff[:3, 3], atol=1e-3)


def test_prepare_subject_on_phantom(tmp_path):
    from types import SimpleNamespace

    from scipy.spatial import cKDTree

    from deepmesh import mesh as M
    from deepmesh.config import load_config
    from deepmesh.real.prepare import prepare_subject
    from deepmesh.synthetic import phantom as P

    rng = np.random.default_rng(3)
    T = 12
    subj = P.sample_subject(rng, T)
    native = SimpleNamespace(height=100, width=100, depth=12, sa_spacing=(1.8, 1.8, 10.0), lax_spacing=(1.8, 1.8, 6.0))
    aff = P.view_affines(rng, subj, native)
    for key, name, shape in (("sa", "sa", (100, 100, 12)), ("la2ch", "la_2ch", (100, 100, 1)), ("la4ch", "la_4ch", (100, 100, 1))):
        grid = P.voxel_grid(aff[key], shape)
        img = np.zeros(shape + (T,), np.float32)
        seg = np.zeros(shape + (T,), np.uint8)
        for t in range(T):
            f = P.contraction(t, T, subj["es_frame"])
            img[..., t], myo = P.render(grid, subj, f)
            s = P.shape_at(subj["ed"], subj["motion"], f)
            p = (grid - subj["centre"]) @ subj["rot"]
            lv = P._inside(p, s["a_endo"], s["b_endo"], s["c_endo"], s["z_base"]) & (p[..., 2] <= s["z_base"])
            seg[..., t] = np.where(myo, 2, np.where(lv, 1, 0))
        nib.save(nib.Nifti1Image(img, aff[key]), tmp_path / f"{name}.nii.gz")
        if name != "la_2ch":
            nib.save(nib.Nifti1Image(seg, aff[key]), tmp_path / f"seg_{name}.nii.gz")

    d = load_config("configs/real.yaml", ["data.height=64", "data.width=64", "data.sa_spacing=[2.5,2.5,2.0]",
                                          "data.lax_spacing=[2.5,2.5,2.0]"]).data
    topo = M.build_topology(48, 20)
    out, qc = prepare_subject(str(tmp_path), d, M.lv_vertices(M.TEMPLATE_SHAPE, topo))
    assert out is not None and qc["reject"] == "", qc
    assert abs(qc["es_frame"] - subj["es_frame"]) <= 1
    assert qc["dice_4ch_sax"] > 0.7

    # the reconstructed ED surface lies on the true myocardial surface
    true_ed = M.lv_vertices(P.shape_at(subj["ed"], subj["motion"], 0.0), M.build_topology(160, 80))
    true_ed = true_ed @ subj["rot"].T + subj["centre"]
    dist = cKDTree(true_ed).query(out["gt_surface_ed"][0])[0]
    assert np.median(dist) < 1.5
    # the template is centred on the LV axis
    tpl_centre = out["template_aligned"].mean(0)
    offset = tpl_centre - subj["centre"]
    axis = subj["rot"][:, 2]
    assert np.linalg.norm(offset - np.dot(offset, axis) * axis) < 4.0


def test_convert_zip_nested(study, tmp_path):
    import shutil
    import zipfile

    from deepmesh.real.dicom import convert_zip

    root = study[0]  # .../train/7/study
    base = root.parent.parent.parent  # folder that contains train/
    inner = tmp_path / "train.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        for f in root.rglob("*.dcm"):
            zf.write(f, f.relative_to(base))
    outer = tmp_path / "bundle.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, "train.zip")
    results = convert_zip(str(outer), str(tmp_path / "native"), str(tmp_path / "work"), log=lambda *_: None)
    assert results == [("dsb_7", results[0][1])] and results[0][1].startswith("ok")
    assert (tmp_path / "native" / "dsb_7" / "sa.nii.gz").exists()
    assert not any((tmp_path / "work").iterdir())  # scratch space cleaned
    shutil.rmtree(tmp_path / "native")
