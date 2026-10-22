"""Stage 1: train the mesh reconstruction module H_D by minimising L_recon (Eq. 11)."""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh.config import base_parser, get_device, load_config  # noqa: E402
from deepmesh.data import ReconstructionDataset, load_template  # noqa: E402
from deepmesh.engine import Logger, fmt, resume, run_epoch, save_checkpoint  # noqa: E402
from deepmesh.pipeline import build_deformation_net, reconstruct, reconstruction_loss  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--resume", action="store_true", help="continue from <models>/reconstruction/last.pth")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device(args.device)
    torch.manual_seed(cfg.data.seed)
    c = cfg.recon

    train_loader = DataLoader(ReconstructionDataset(os.path.join(cfg.data.root, "train")), c.batch_size,
                              shuffle=True, num_workers=c.num_workers, drop_last=True)
    val_loader = DataLoader(ReconstructionDataset(os.path.join(cfg.data.root, "val")), c.batch_size,
                            num_workers=c.num_workers)
    _, edges = load_template(cfg.data.root)
    edges = edges.to(device)

    net = build_deformation_net(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=c.lr)

    def loss_fn(model, batch):
        verts, field = reconstruct(model, batch, cfg)
        return reconstruction_loss(verts, field, batch, edges, cfg)

    model_dir = os.path.join(cfg.output.models, "reconstruction")
    first, best = 1, float("inf")
    if args.resume:
        first, best = resume(os.path.join(model_dir, "last.pth"), net, optimizer, device)
    logger = Logger(os.path.join(cfg.output.runs, "reconstruction"), first)
    print(f"device={device} params={sum(p.numel() for p in net.parameters()):,} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    for epoch in range(first, c.epochs + 1):
        start = time.time()
        train = run_epoch(net, train_loader, loss_fn, device, optimizer, desc=f"train {epoch}")
        val = run_epoch(net, val_loader, loss_fn, device, desc=f"val {epoch}")
        logger.log(epoch, train, val, time.time() - start)
        print(f"[{epoch:03d}] {time.time() - start:5.1f}s | train {fmt(train)} | val {fmt(val)}")
        if val["total"] < best:  # the reference code keeps the model with the lowest validation loss
            best = val["total"]
            save_checkpoint(os.path.join(model_dir, "best.pth"), net, epoch, best, args.config)
        save_checkpoint(os.path.join(model_dir, "last.pth"), net, epoch, val["total"], args.config, optimizer, best)
    logger.close()
    print(f"best val loss {best:.4f} -> {model_dir}/best.pth")


if __name__ == "__main__":
    main()
