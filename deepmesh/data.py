"""Datasets for the two training stages (see scripts/generate_synthetic.py for the layout)."""
import json
import os

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from torch.utils.data import Dataset

from deepmesh.mesh import unique_edges

VIEWS = ("sa", "2ch", "4ch")


def load_template(root):
    t = np.load(os.path.join(root, "template.npz"))
    faces = torch.as_tensor(t["faces"], dtype=torch.long)
    edges = torch.as_tensor(unique_edges(t["faces"]), dtype=torch.long)
    return faces, edges


def contour(mask):
    """Myocardium contour pixels of a 2D mask (or a stack along the last axis)."""
    if mask.ndim == 3:
        return np.stack([contour(mask[..., k]) for k in range(mask.shape[-1])], -1)
    mask = mask.astype(bool)
    return (mask & ~binary_erosion(mask)).astype(np.float32)


class Subject:
    """Lazy access to one subject folder."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self._nii = {}

    def _load(self, name):
        if name not in self._nii:
            self._nii[name] = nib.load(os.path.join(self.path, f"{name}.nii.gz"))
        return self._nii[name]

    def frame(self, view, t):
        """Image of one frame normalised to [0, 1]: (H, W, D) for SAX, (H, W) for LAX."""
        data = np.asanyarray(self._load(view).dataobj)[..., t].astype(np.float32) / 255.0
        return data if view == "sa" else data[..., 0]

    def label(self, view, t):
        if not os.path.exists(os.path.join(self.path, f"{view}_myo.nii.gz")):
            # view without segmentation (e.g. 2CH of real data): empty label
            shape = self._load(view).shape[:3]
            return np.zeros(shape if view == "sa" else shape[:2], np.uint8)
        data = np.asanyarray(self._load(f"{view}_myo").dataobj)[..., t]
        return data if view == "sa" else data[..., 0]

    def affines(self):
        out = {}
        for v in VIEWS:
            A = self._load(v).affine.astype(np.float64)
            out[v] = torch.as_tensor(A, dtype=torch.float32)
            out[f"{v}_inv"] = torch.as_tensor(np.linalg.inv(A), dtype=torch.float32)
        return out

    def array(self, name):
        p = os.path.join(self.path, name)
        if not os.path.exists(p):
            return None
        if name.endswith(".npz"):
            return np.load(p)["verts"]
        return np.load(p)


def _images(subj, t):
    """Network inputs: SAX as (D, H, W) so slices are channels, LAX as (1, H, W)."""
    return {"sa": torch.from_numpy(subj.frame("sa", t).transpose(2, 0, 1).copy()),
            "2ch": torch.from_numpy(subj.frame("2ch", t))[None],
            "4ch": torch.from_numpy(subj.frame("4ch", t))[None]}


def list_subjects(split_dir):
    if not os.path.isdir(split_dir):  # e.g. a split that received no subject
        return []
    return [Subject(os.path.join(split_dir, n)) for n in sorted(os.listdir(split_dir))]


class ReconstructionDataset(Dataset):
    """ED-frame inputs and targets for the mesh reconstruction module (Sec. III-A)."""

    def __init__(self, split_dir):
        self.subjects = list_subjects(split_dir)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        s = self.subjects[idx]
        t = s.meta["ed_frame"]
        return dict(
            name=s.name, images=_images(s, t), affines=s.affines(),
            contour_sa=torch.from_numpy(contour(s.label("sa", t))),        # (H, W, D) all slices
            contour_2ch=torch.from_numpy(contour(s.label("2ch", t))),
            contour_4ch=torch.from_numpy(contour(s.label("4ch", t))),
            template=torch.from_numpy(s.array("template_aligned.npy")),
            gt_surface=torch.from_numpy(s.array("gt_surface_ed.npz")),
        )


class MotionDataset(Dataset):
    """ED + frame-t inputs and targets for the mesh motion module (Sec. III-B).

    frame: None draws t uniformly over the cycle (training); "es" uses the ES
    frame (validation/test, the frame with the largest deformation); an int fixes t.
    """

    def __init__(self, split_dir, sa_slices, frame=None, use_gt_ed_mesh=False):
        self.subjects = list_subjects(split_dir)
        self.sa_slices = list(sa_slices)
        self.frame = frame
        self.use_gt_ed_mesh = use_gt_ed_mesh

    def __len__(self):
        return len(self.subjects)

    def ed_vertices(self, s):
        pred = s.array("pred_ed_vertices.npy")
        if pred is None or self.use_gt_ed_mesh:
            gt = s.array("gt_vertices.npy")
            if gt is None:
                raise FileNotFoundError(f"{s.path}: run scripts/export_ed_meshes.py first")
            return gt[s.meta["ed_frame"]]
        return pred

    def __getitem__(self, idx):
        s = self.subjects[idx]
        if self.frame is None:
            # torch RNG: correctly re-seeded in every DataLoader worker
            t = int(torch.randint(s.meta["n_frames"], ()))
        elif self.frame == "es":
            t = s.meta["es_frame"]
        else:
            t = int(self.frame)
        # same keys in every item of a batch: the ES surface only in "es" mode
        return self.get(idx, t, with_es_surface=self.frame == "es")

    def get(self, idx, t, with_es_surface=False):
        s = self.subjects[idx]
        ed = s.meta["ed_frame"]
        gt_vertices = s.array("gt_vertices.npy")
        item = dict(
            name=s.name, frame=t, images_ed=_images(s, ed), images_t=_images(s, t), affines=s.affines(),
            contour_sa=torch.from_numpy(contour(s.label("sa", t)[..., self.sa_slices])),  # (H, W, 9)
            contour_2ch=torch.from_numpy(contour(s.label("2ch", t))),
            contour_4ch=torch.from_numpy(contour(s.label("4ch", t))),
            ed_verts=torch.from_numpy(np.asarray(self.ed_vertices(s), np.float32)),
        )
        if gt_vertices is not None:
            item["gt_verts_t"] = torch.from_numpy(gt_vertices[t])
            item["gt_verts_ed"] = torch.from_numpy(gt_vertices[ed])
        if with_es_surface:
            item["gt_surface"] = torch.from_numpy(s.array("gt_surface_es.npz"))
        return item
