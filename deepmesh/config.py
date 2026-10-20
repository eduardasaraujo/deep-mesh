"""Configuration loading (YAML + command-line overrides)."""
import argparse
import copy
from types import SimpleNamespace

import torch
import yaml


def _to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d


def _set_by_path(d, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    if keys[-1] not in d:
        raise KeyError(f"unknown config key: {path}")
    d[keys[-1]] = _parse_value(value)


def _parse_value(text):
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:  # YAML 1.1 reads "1e-3" as a string
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path, overrides=()):
    """Load a YAML config. `overrides` is a list of 'section.key=value' strings."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = copy.deepcopy(raw)
    for item in overrides:
        key, value = item.split("=", 1)
        _set_by_path(raw, key, value)
    return _to_namespace(raw)


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set recon.epochs=5 data.root=data/x")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    return p


def get_device(name="auto"):
    if name == "auto":
        # ROCm builds of PyTorch also expose AMD GPUs through torch.cuda
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
