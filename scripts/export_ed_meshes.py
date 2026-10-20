"""Run the trained reconstruction module on every subject and save the ED mesh.

The motion module (stage 2) starts from these reconstructed meshes, as in the
paper's inference pipeline. Writes <subject>/pred_ed_vertices.npy.
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import default_collate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh.config import base_parser, get_device, load_config  # noqa: E402
from deepmesh.data import ReconstructionDataset  # noqa: E402
from deepmesh.engine import load_checkpoint  # noqa: E402
from deepmesh.pipeline import build_deformation_net, reconstruct, to_device  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device(args.device)

    net = build_deformation_net(cfg).to(device)
    load_checkpoint(args.checkpoint or os.path.join(cfg.output.models, "reconstruction", "best.pth"), net, device)
    net.eval()
    for split in ("train", "val", "test"):
        dataset = ReconstructionDataset(os.path.join(cfg.data.root, split))
        for idx in range(len(dataset)):
            batch = to_device(default_collate([dataset[idx]]), device)
            with torch.no_grad():
                verts = reconstruct(net, batch, cfg)[0][0].cpu().numpy()
            np.save(os.path.join(dataset.subjects[idx].path, "pred_ed_vertices.npy"), verts.astype(np.float32))
        print(f"{split}: {len(dataset)} ED meshes exported")


if __name__ == "__main__":
    main()
