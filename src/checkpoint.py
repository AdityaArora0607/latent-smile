"""Checkpoint save/load."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from .models import VAE, build_model
from .utils import get_device


def save_checkpoint(path, model, optimizer, epoch, cfg, best_val=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": cfg,
        "best_val": best_val,
    }, tmp)
    os.replace(tmp, path)


def resolve_checkpoint(path) -> Path:
    path = Path(path)
    if path.is_dir():
        for name in ("best.pt", "last.pt"):
            if (path / name).exists():
                return path / name
        raise FileNotFoundError(f"no best.pt or last.pt in {path}")
    return path


def load_model(path, device=None) -> tuple[VAE, dict]:
    device = torch.device(device) if device is not None else get_device()
    ckpt = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg
