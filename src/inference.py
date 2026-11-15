"""Deterministic smile editing for both approaches.

Both approaches set an *absolute* smile coordinate, so two different inputs edited at
the same ``t`` end at the same smile coordinate (same ``z_smile`` for supervised models,
same projection onto ``v`` for discovered directions).
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.checkpoint import load_model
from src.utils import get_device

DIRECTION_NAMES = ("probe", "mean_diff", "coordinate")


def _run_dir(run) -> Path:
    p = Path(run)
    return p if p.is_dir() else p.parent


class SmileEditor:
    """Edit smile intensity of [B,3,S,S] images in [-1, 1].

    t = 0 is level 1 (least smiling), t = 1 is level 5; values outside [0, 1] extrapolate.
    """

    def __init__(self, model, kind: str, device=None, direction_entry: dict | None = None,
                 direction_name: str | None = None):
        if kind not in ("supervised", "discovered"):
            raise ValueError(f"unknown kind {kind!r}")
        self.model = model
        self.kind = kind
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.image_size = int(model.image_size)
        self.direction_name = direction_name
        self.model.eval()
        if kind == "supervised":
            m = float(model.smile_prior_mean)
            self.lo, self.hi = -m, m
            self.v = None
        else:
            if direction_entry is None:
                raise ValueError("a discovered editor needs a direction entry {'v','p05','p95'}")
            self.v = torch.as_tensor(direction_entry["v"], dtype=torch.float32, device=self.device)
            self.v = self.v / self.v.norm().clamp_min(1e-12)
            self.lo = float(direction_entry["p05"])
            self.hi = float(direction_entry["p95"])

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, run, device=None, direction: str = "probe", direction_entry: dict | None = None):
        model, cfg = load_model(run, device)
        device = next(model.parameters()).device
        mtype = cfg["model"]["type"]
        if mtype == "supervised":
            return cls(model, "supervised", device)
        if mtype != "vae":
            raise ValueError(f"unknown model type {mtype!r}")
        if direction_entry is None:
            path = _run_dir(run) / "directions.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"No directions.pt found at {path}. A plain VAE has no built-in smile axis; "
                    f"run `python -m src.discover_smile --run {_run_dir(run)}` first."
                )
            if direction not in DIRECTION_NAMES:
                raise ValueError(f"direction must be one of {DIRECTION_NAMES}, got {direction!r}")
            directions = torch.load(path, map_location="cpu", weights_only=False)
            direction_entry = directions[direction]
        return cls(model, "discovered", device, direction_entry, direction)

    # ------------------------------------------------------------------ helpers
    def _to_dev(self, x):
        return x.to(self.device, dtype=torch.float32)

    @property
    def target_range(self) -> tuple[float, float]:
        return self.lo, self.hi

    def _targets(self, t, B: int) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32, device=self.device)
        if t.ndim == 0:
            t = t.expand(B)
        if t.shape != (B,):
            raise ValueError(f"t must be a float or a [B] tensor, got shape {tuple(t.shape)}")
        return self.lo + t * (self.hi - self.lo)

    def _edit_z(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "supervised":
            z = z.clone()
            z[:, -1] = target
            return z
        proj = z @ self.v
        return z + (target - proj)[:, None] * self.v[None, :]

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def encode(self, x):
        self.model.eval()
        mu, _ = self.model.encode(self._to_dev(x))
        return mu

    @torch.no_grad()
    def reconstruct(self, x):
        return self.model.decode(self.encode(x))

    @torch.no_grad()
    def smile_coordinate(self, x):
        z = self.encode(x)
        return z[:, -1] if self.kind == "supervised" else z @ self.v

    @torch.no_grad()
    def edit_at(self, x, t):
        z = self.encode(x)
        return self.model.decode(self._edit_z(z, self._targets(t, z.shape[0])))

    @torch.no_grad()
    def edit_levels(self, x, n: int = 5):
        z = self.encode(x)
        B = z.shape[0]
        ts = torch.linspace(0, 1, n, device=self.device)
        outs = [self.model.decode(self._edit_z(z, self._targets(float(t), B))) for t in ts]
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def content_code(self, x):
        z = self.encode(x)
        if self.kind == "supervised":
            return z[:, :-1]
        return z - (z @ self.v)[:, None] * self.v[None, :]
