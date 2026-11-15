"""Small shared helpers: device selection and seeding."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """MPS if available, else CUDA, else CPU. ``LATENT_SMILE_DEVICE`` overrides."""
    override = os.environ.get("LATENT_SMILE_DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (all devices)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
