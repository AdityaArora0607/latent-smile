"""Approach A: VAE whose last latent dim is a supervised smile axis."""
from __future__ import annotations

import torch
from torch import nn

from .vae import VAE


class GradReverse(torch.autograd.Function):
    """Identity forward; multiplies the gradient by ``-lambd`` backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class SupervisedVAE(VAE):
    """z = [z_content (D-1 dims), z_smile (last dim)]. The decoder sees the full z."""

    def __init__(self, image_size=64, latent_dim=64, base_channels=32,
                 smile_prior_mean: float = 2.0, adv_hidden: int = 128):
        super().__init__(image_size, latent_dim, base_channels)
        if latent_dim < 2:
            raise ValueError("SupervisedVAE needs latent_dim >= 2")
        self.smile_prior_mean = float(smile_prior_mean)
        # Always built (even with adv_weight == 0) so checkpoints share one layout.
        self.adversary = nn.Sequential(
            nn.Linear(latent_dim - 1, adv_hidden), nn.SiLU(),
            nn.Linear(adv_hidden, adv_hidden), nn.SiLU(),
            nn.Linear(adv_hidden, 1),
        )

    def split(self, z):
        return z[:, :-1], z[:, -1]

    def adversary_logits(self, z_content, lambd: float = 1.0):
        return self.adversary(grad_reverse(z_content, lambd)).squeeze(-1)
