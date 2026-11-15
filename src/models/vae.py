"""Convolutional VAE backbone (encoder, decoder, VAE)."""
from __future__ import annotations

import math

import torch
from torch import nn

MAX_CHANNELS = 256
FEATURE_SIZE = 4  # spatial size of the encoder's last feature map


def _norm(c: int) -> nn.GroupNorm:
    groups = next(g for g in (8, 4, 2, 1) if c % g == 0)
    return nn.GroupNorm(groups, c)


def num_blocks(image_size: int) -> int:
    """Stride-2 blocks needed to go from ``image_size`` down to 4x4 (4 at 64, 5 at 128)."""
    n = int(round(math.log2(image_size // FEATURE_SIZE)))
    if FEATURE_SIZE * 2**n != image_size or n < 1:
        raise ValueError(f"image_size must be 4 * 2^k, got {image_size}")
    return n


def channel_plan(image_size: int, base_channels: int) -> list[int]:
    """Encoder channels per block: base doubling, capped at 256."""
    return [min(base_channels * 2**i, MAX_CHANNELS) for i in range(num_blocks(image_size))]


class Encoder(nn.Module):
    def __init__(self, image_size: int, latent_dim: int, base_channels: int):
        super().__init__()
        chans = channel_plan(image_size, base_channels)
        layers, c_in = [], 3
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 4, stride=2, padding=1), _norm(c), nn.SiLU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.flat_dim = c_in * FEATURE_SIZE * FEATURE_SIZE
        self.fc = nn.Linear(self.flat_dim, 2 * latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu, logvar = self.fc(h).chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, image_size: int, latent_dim: int, base_channels: int):
        super().__init__()
        chans = channel_plan(image_size, base_channels)[::-1]  # 256 ... base
        self.c0 = chans[0]
        self.fc = nn.Linear(latent_dim, self.c0 * FEATURE_SIZE * FEATURE_SIZE)
        layers = [nn.SiLU()]
        for i, c_in_block in enumerate(chans):
            c_out = chans[i + 1] if i + 1 < len(chans) else chans[-1]
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c_in_block, c_out, 3, padding=1),
                _norm(c_out),
                nn.SiLU(),
            ]
        self.up = nn.Sequential(*layers)
        self.out = nn.Conv2d(chans[-1], 3, 3, padding=1)

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], self.c0, FEATURE_SIZE, FEATURE_SIZE)
        return torch.tanh(self.out(self.up(h)))


class VAE(nn.Module):
    def __init__(self, image_size: int = 64, latent_dim: int = 64, base_channels: int = 32):
        super().__init__()
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.encoder = Encoder(self.image_size, self.latent_dim, self.base_channels)
        self.decoder = Decoder(self.image_size, self.latent_dim, self.base_channels)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {"x_hat": self.decode(z), "mu": mu, "logvar": logvar, "z": z}
