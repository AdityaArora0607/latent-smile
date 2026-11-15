"""YAML config loading with defaults and dotted-key overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": "baseline",
    "model": {
        "type": "vae",
        "image_size": 64,
        "latent_dim": 64,
        "base_channels": 32,
        "smile_prior_mean": 2.0,
        "adv_hidden": 128,
    },
    "loss": {
        "sigma": 1.0,
        "beta": 1.0,
        "kl_warmup_epochs": 5,
        "adv_weight": 1.0,
        "adv_warmup_epochs": 5,
    },
    "train": {
        "data_root": "data/celeba_64",
        "out_dir": "runs",
        "epochs": 30,
        "batch_size": 128,
        "lr": 0.001,
        "seed": 0,
        "subset": None,
        "num_workers": 0,
        "flip": True,
    },
}


def _merge(base: dict, new: dict, path: str = "") -> dict:
    for k, v in new.items():
        if k not in base:
            raise KeyError(f"unknown config key: {path}{k}")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f"config key {path}{k} must be a mapping")
            _merge(base[k], v, f"{path}{k}.")
        else:
            base[k] = v
    return base


def fill_defaults(cfg: dict | None) -> dict:
    """Return a fresh, complete config (defaults filled in, input not mutated)."""
    out = copy.deepcopy(DEFAULTS)
    _merge(out, copy.deepcopy(cfg or {}))
    if out["model"]["type"] not in ("vae", "supervised"):
        raise ValueError(f"model.type must be 'vae' or 'supervised', got {out['model']['type']!r}")
    return out


def apply_overrides(cfg: dict, overrides: dict | None) -> dict:
    """Apply dotted-key overrides (e.g. ``{"train.epochs": 2}``) to a complete config."""
    cfg = fill_defaults(cfg)
    for key, value in (overrides or {}).items():
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"unknown config key: {key}")
            node = node[p]
        if parts[-1] not in node or isinstance(node[parts[-1]], dict):
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = value
    return cfg


def load_config(path, overrides: dict | None = None) -> dict:
    with open(Path(path)) as f:
        raw = yaml.safe_load(f) or {}
    return apply_overrides(fill_defaults(raw), overrides)
