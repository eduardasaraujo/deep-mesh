"""Generate the synthetic dataset in the on-disk format read by deepmesh.data.

Layout:
    <root>/template.npz                     template mesh (local frame) + faces
    <root>/{train,val,test}/<subject>/
        sa.nii.gz, 2ch.nii.gz, 4ch.nii.gz            uint8 cines (H, W, D|1, T)
        sa_myo.nii.gz, 2ch_myo.nii.gz, 4ch_myo.nii.gz myocardium labels
        template_aligned.npy                 template moved to subject space (N, 3)
        gt_surface_ed.npz, gt_surface_es.npz  GT meshes (different topology)
        gt_vertices.npy                      GT vertices in template topology (T, N, 3)
        meta.json                            ED/ES frame indices and phantom parameters
"""
import json
import os
import sys

import nibabel as nib
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh import mesh as M  # noqa: E402
from deepmesh.config import base_parser, load_config  # noqa: E402
from deepmesh.synthetic.phantom import generate_subject  # noqa: E402

VIEW_FILES = {"sa": "sa", "la2ch": "2ch", "la4ch": "4ch"}


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_subject(path, out, gt_faces):
    os.makedirs(path, exist_ok=True)
    for view, name in VIEW_FILES.items():
        affine = out["affines"][view]
        nib.save(nib.Nifti1Image(out["images"][view], affine), os.path.join(path, f"{name}.nii.gz"))
        nib.save(nib.Nifti1Image(out["labels"][view], affine), os.path.join(path, f"{name}_myo.nii.gz"))
    np.save(os.path.join(path, "template_aligned.npy"), out["template_aligned"])
    np.save(os.path.join(path, "gt_vertices.npy"), out["gt_vertices"])
    for key in ("gt_surface_ed", "gt_surface_es"):
        np.savez(os.path.join(path, f"{key}.npz"), verts=out[key], faces=gt_faces)
    subj = out["subject"]
    meta = dict(ed_frame=0, es_frame=subj["es_frame"], n_frames=subj["n_frames"],
                phantom=_jsonable({k: subj[k] for k in ("ed", "motion", "rot", "centre")}))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg.data
    rng = np.random.default_rng(d.seed)

    topo = M.build_topology(d.mesh_around, d.mesh_long)
    template_local = M.lv_vertices(M.TEMPLATE_SHAPE, topo)
    # GT meshes use a different resolution, like marching-cubes meshes in the paper
    gt_topo = M.build_topology(int(round(d.mesh_around * 1.25)), int(round(d.mesh_long * 1.25)))

    os.makedirs(d.root, exist_ok=True)
    np.savez(os.path.join(d.root, "template.npz"), verts=template_local.astype(np.float32),
             faces=topo.faces, n_around=d.mesh_around, n_long=d.mesh_long)
    print(f"template: {topo.n_verts} vertices, {len(topo.faces)} faces")

    for split, n in (("train", d.n_train), ("val", d.n_val), ("test", d.n_test)):
        for idx in tqdm(range(n), desc=split):
            out = generate_subject(rng, d, topo, gt_topo, template_local)
            write_subject(os.path.join(d.root, split, f"subject_{idx:04d}"), out, gt_topo.faces)


if __name__ == "__main__":
    main()
