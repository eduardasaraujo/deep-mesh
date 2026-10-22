"""Evaluate mesh motion tracking (Table II of the paper) and predict whole cycles.

* ES frame: HD / BoundF on SAX, 2CH, 4CH, surface distance to the GT ES mesh and,
  on synthetic data, vertex-wise error. "no motion" (ED mesh used as the ES mesh)
  is reported as a baseline.
* Full cycle: the ED mesh is deformed to every frame t (Fig. 7 / Fig. 8). The
  predicted sequences are saved for visualisation, together with the per-frame
  vertex error and the "motion error" |(V_t - V_0) - (V*_t - V*_0)| that ignores
  the error of the reconstructed ED mesh.
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import default_collate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh import metrics as Mt  # noqa: E402
from deepmesh.config import base_parser, get_device, load_config  # noqa: E402
from deepmesh.data import MotionDataset  # noqa: E402
from deepmesh.engine import load_checkpoint  # noqa: E402
from deepmesh.pipeline import build_motion_net, shape_views, to_device, track  # noqa: E402


def evaluate(verts, item, cfg):
    d, e = cfg.data, cfg.eval
    affines = {k: v.numpy().astype(np.float64) for k, v in item["affines"].items()}
    contours = {"sa": item["contour_sa"].numpy(), "2ch": item["contour_2ch"].numpy(),
                "4ch": item["contour_4ch"].numpy()}
    row = Mt.contour_metrics(verts, affines, contours, d.sa_slices, e.sa_eval_slices,
                             d.sa_spacing[0], d.lax_spacing[0], e.slice_band, shape_views(cfg))
    row["surface_dist"] = Mt.surface_distance(verts, item["gt_surface"].numpy())
    row["vertex_err"] = Mt.vertex_error(verts, item["gt_verts_t"].numpy()) if "gt_verts_t" in item else None
    return row


def predict(net, item, cfg, device):
    batch = to_device(default_collate([item]), device)
    with torch.no_grad():
        return track(net, batch, cfg)[0][0].cpu().numpy().astype(np.float64)


def main():
    p = base_parser(__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--no-cycle", action="store_true", help="only evaluate the ES frame")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device(args.device)

    net = build_motion_net(cfg).to(device)
    load_checkpoint(args.checkpoint or os.path.join(cfg.output.models, "motion", "best.pth"), net, device)
    net.eval()

    out_dir = os.path.join(cfg.output.results, "motion")
    os.makedirs(os.path.join(out_dir, "sequences"), exist_ok=True)
    dataset = MotionDataset(os.path.join(cfg.data.root, args.split), cfg.data.sa_slices, "es",
                            cfg.motion.use_gt_ed_mesh)
    rows = {"deepmesh": [], "no_motion": []}
    cycle = {}
    for idx, subj in enumerate(dataset.subjects):
        item = dataset[idx]
        rows["deepmesh"].append(dict(name=subj.name, **evaluate(predict(net, item, cfg, device), item, cfg)))
        rows["no_motion"].append(dict(name=subj.name, **evaluate(item["ed_verts"].numpy(), item, cfg)))
        if args.no_cycle:
            continue
        seq, err, motion_err = [], [], []
        for t in range(subj.meta["n_frames"]):
            it = dataset.get(idx, t)
            v = predict(net, it, cfg, device)
            seq.append(v)
            if "gt_verts_t" in it:
                gt_t, gt_0, v_0 = it["gt_verts_t"].numpy(), it["gt_verts_ed"].numpy(), it["ed_verts"].numpy()
                err.append(Mt.vertex_error(v, gt_t))
                motion_err.append(Mt.vertex_error(v - v_0, gt_t - gt_0))
        np.save(os.path.join(out_dir, "sequences", f"{subj.name}.npy"), np.stack(seq).astype(np.float32))
        cycle[subj.name] = dict(es_frame=subj.meta["es_frame"], vertex_err=err, motion_err=motion_err)

    summary = {k: Mt.summarize(v) for k, v in rows.items()}
    with open(os.path.join(out_dir, f"{args.split}_metrics.json"), "w") as f:
        json.dump(dict(summary=summary, subjects=rows, cycle=cycle), f, indent=1)
    print("ES frame. HD in mm, BoundF in %, surface distance and vertex error in mm; mean(std)")
    Mt.print_table("sem movimento (malha ED usada como ES)", summary["no_motion"])
    Mt.print_table("DeepMesh (rastreamento ED -> ES)", summary["deepmesh"])
    if cycle:
        me = np.array([c["motion_err"] for c in cycle.values() if c["motion_err"]])
        if me.size:
            print(f"\nmotion error over the cycle: mean {me.mean():.2f} mm, worst frame {me.mean(0).max():.2f} mm")
    print(f"saved {out_dir}/{args.split}_metrics.json and sequences/")


if __name__ == "__main__":
    main()
