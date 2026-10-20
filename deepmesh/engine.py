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


def save_checkpoint(path, net, epoch, val_loss, config_path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model=net.state_dict(), epoch=epoch, val_loss=val_loss, config=config_path), path)


def load_checkpoint(path, net, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["model"])
    return ckpt


class Logger:
    """TensorBoard scalars plus a CSV history (one row per epoch)."""

    def __init__(self, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(run_dir)
        self.csv_path = os.path.join(run_dir, "history.csv")
        self.rows = []

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
