"""Generic training loop, checkpointing and logging."""
import csv
import os
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deepmesh.pipeline import to_device


def run_epoch(net, loader, loss_fn, device, optimizer=None, desc=""):
    """One pass over `loader`; trains when an optimizer is given. Returns mean losses."""
    training = optimizer is not None
    net.train(training)
    sums, n = defaultdict(float), 0
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = to_device(batch, device)
            total, terms = loss_fn(net, batch)
            if training:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
            bs = len(batch["name"])
            n += bs
            sums["total"] += total.item() * bs
            for k, v in terms.items():
                sums[k] += v.item() * bs
    return {k: v / n for k, v in sums.items()}


def save_checkpoint(path, net, epoch, val_loss, config_path, optimizer=None, best=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = dict(model=net.state_dict(), epoch=epoch, val_loss=val_loss, config=config_path, best=best)
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)  # never leave a half-written checkpoint (e.g. Colab disconnects)


def load_checkpoint(path, net, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["model"])
    return ckpt


def resume(path, net, optimizer, device):
    """Continue from `path` if it exists; returns (first epoch to run, best validation loss)."""
    if not os.path.exists(path):
        return 1, float("inf")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    best = ckpt.get("best")
    print(f"resuming after epoch {ckpt['epoch']} from {path}")
    return ckpt["epoch"] + 1, float("inf") if best is None else best


class Logger:
    """TensorBoard scalars plus a CSV history (one row per epoch)."""

    def __init__(self, run_dir, first_epoch=1):
        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(run_dir)
        self.csv_path = os.path.join(run_dir, "history.csv")
        self.rows = []
        if first_epoch > 1 and os.path.exists(self.csv_path):  # resumed run: keep earlier epochs
            with open(self.csv_path) as f:
                self.rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < first_epoch]

    def log(self, epoch, train, val, seconds):
        row = {"epoch": epoch, "seconds": round(seconds, 1)}
        for split, values in (("train", train), ("val", val)):
            for k, v in values.items():
                self.writer.add_scalar(f"{k}/{split}", v, epoch)
                row[f"{split}_{k}"] = v
        self.rows.append(row)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(self.rows)

    def close(self):
        self.writer.close()


def fmt(values):
    return " ".join(f"{k}={v:.4f}" for k, v in values.items())
