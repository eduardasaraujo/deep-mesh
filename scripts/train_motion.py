"""Stage 2: train the mesh motion module H_M by minimising L_motion (Eq. 12).

Needs the reconstructed ED meshes (scripts/export_ed_meshes.py), unless
motion.use_gt_ed_mesh=true. For each training iteration the frame t is drawn at
random from the cardiac cycle; validation uses the ES frame.
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh.config import base_parser, get_device, load_config  # noqa: E402
from deepmesh.data import MotionDataset, load_template  # noqa: E402
from deepmesh.engine import Logger, fmt, resume, run_epoch, save_checkpoint  # noqa: E402
from deepmesh.pipeline import build_motion_net, motion_loss, track  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--resume", action="store_true", help="continue from <models>/motion/last.pth")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device(args.device)
    torch.manual_seed(cfg.data.seed)
    c, d = cfg.motion, cfg.data

    def dataset(split, frame):
        return MotionDataset(os.path.join(d.root, split), d.sa_slices, frame, c.use_gt_ed_mesh,
                             es_prob=getattr(c, "es_prob", 0.0))

    train_loader = DataLoader(dataset("train", None), c.batch_size, shuffle=True,
                              num_workers=c.num_workers, drop_last=True)
    val_loader = DataLoader(dataset("val", "es"), c.batch_size, num_workers=c.num_workers)
    _, edges = load_template(d.root)
    edges = edges.to(device)

    net = build_motion_net(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=c.lr)

    def loss_fn(model, batch):
        verts, field = track(model, batch, cfg)
        return motion_loss(verts, field, batch, edges, cfg)

    model_dir = os.path.join(cfg.output.models, "motion")
    first, best = 1, float("inf")
    if args.resume:
        first, best = resume(os.path.join(model_dir, "last.pth"), net, optimizer, device)
    logger = Logger(os.path.join(cfg.output.runs, "motion"), first)
    print(f"device={device} params={sum(p.numel() for p in net.parameters()):,} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} gt_ed_mesh={c.use_gt_ed_mesh}")
    for epoch in range(first, c.epochs + 1):
        start = time.time()
        train = run_epoch(net, train_loader, loss_fn, device, optimizer, desc=f"train {epoch}")
        val = run_epoch(net, val_loader, loss_fn, device, desc=f"val {epoch}")
        logger.log(epoch, train, val, time.time() - start)
        print(f"[{epoch:03d}] {time.time() - start:5.1f}s | train {fmt(train)} | val {fmt(val)}")
        if val["total"] < best:
            best = val["total"]
            save_checkpoint(os.path.join(model_dir, "best.pth"), net, epoch, best, args.config)
        save_checkpoint(os.path.join(model_dir, "last.pth"), net, epoch, val["total"], args.config, optimizer, best)
    logger.close()
    print(f"best val loss {best:.4f} -> {model_dir}/best.pth")


if __name__ == "__main__":
    main()
