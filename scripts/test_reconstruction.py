"""Evaluate the reconstructed ED meshes on the test set (Table I of the paper).

The template aligned from the image headers only (before H_D) is reported as a
baseline, which shows what the deformation network adds.
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
from deepmesh.data import ReconstructionDataset  # noqa: E402
from deepmesh.engine import load_checkpoint  # noqa: E402
from deepmesh.pipeline import build_deformation_net, reconstruct, shape_views, to_device  # noqa: E402



def evaluate(verts, item, cfg):
    d, e = cfg.data, cfg.eval
    affines = {k: v.numpy().astype(np.float64) for k, v in item["affines"].items()}
    contours = {"sa": item["contour_sa"].numpy()[..., d.sa_slices],
                "2ch": item["contour_2ch"].numpy(), "4ch": item["contour_4ch"].numpy()}
    row = Mt.contour_metrics(verts, affines, contours, d.sa_slices, e.sa_eval_slices,
                             d.sa_spacing[0], d.lax_spacing[0], e.slice_band, shape_views(cfg))
    row["surface_dist"] = Mt.surface_distance(verts, item["gt_surface"].numpy())
    return row


def main():
    p = base_parser(__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device(args.device)

    net = build_deformation_net(cfg).to(device)
    ckpt = args.checkpoint or os.path.join(cfg.output.models, "reconstruction", "best.pth")
    load_checkpoint(ckpt, net, device)
    net.eval()

    dataset = ReconstructionDataset(os.path.join(cfg.data.root, args.split))
    rows = {"deepmesh": [], "template_only": []}
    for idx in range(len(dataset)):
        item = dataset[idx]
        batch = to_device(default_collate([item]), device)
        with torch.no_grad():
            verts = reconstruct(net, batch, cfg)[0][0].cpu().numpy().astype(np.float64)
        gt = dataset.subjects[idx].array("gt_vertices.npy")
        for name, v in (("deepmesh", verts), ("template_only", item["template"].numpy())):
            row = dict(name=item["name"], **evaluate(v, item, cfg))
            row["vertex_err"] = Mt.vertex_error(v, gt[0]) if gt is not None else None
            rows[name].append(row)

    out_dir = os.path.join(cfg.output.results, "reconstruction")
    os.makedirs(out_dir, exist_ok=True)
    summary = {k: Mt.summarize(v) for k, v in rows.items()}
    with open(os.path.join(out_dir, f"{args.split}_metrics.json"), "w") as f:
        json.dump(dict(summary=summary, subjects=rows), f, indent=1)
    print("HD in mm, BoundF in %, surface distance and vertex error in mm; mean(std)")
    Mt.print_table("template alinhado (sem rede)", summary["template_only"])
    Mt.print_table("DeepMesh (reconstrução ED)", summary["deepmesh"])
    print(f"\nsaved {out_dir}/{args.split}_metrics.json")


if __name__ == "__main__":
    main()
