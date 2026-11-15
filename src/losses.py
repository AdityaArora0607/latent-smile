"""Losses: fixed-sigma Gaussian likelihood + KL (+ adversary for approach A)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _recon(x_hat, x, sigma):
    """SSE / (2 sigma^2) summed per image, averaged over the batch."""
    return ((x_hat - x) ** 2).flatten(1).sum(1).mean() / (2.0 * sigma ** 2)


def kl_per_dim_gaussian(mu, logvar, prior_mu=None):
    """KL(N(mu, exp(logvar)) || N(prior_mu, 1)) per dim, averaged over the batch -> [D]."""
    if prior_mu is None:
        prior_mu = torch.zeros_like(mu)
    kl = 0.5 * ((mu - prior_mu) ** 2 + logvar.exp() - 1.0 - logvar)
    return kl.mean(0)


def vae_loss(out, x, sigma, beta):
    recon = _recon(out["x_hat"], x, sigma)
    kl_per_dim = kl_per_dim_gaussian(out["mu"], out["logvar"])
    kl = kl_per_dim.sum()
    return {"loss": recon + beta * kl, "recon": recon, "kl": kl, "kl_per_dim": kl_per_dim}


def supervised_loss(model, out, x, smiling, sigma, beta, adv_weight, lambd):
    """smiling: float [B] in {0,1}. z_smile (last dim) has prior N(+m,1) if smiling else N(-m,1)."""
    smiling = smiling.to(out["mu"].dtype)
    m = model.smile_prior_mean
    prior_mu = torch.zeros_like(out["mu"])
    prior_mu[:, -1] = (2.0 * smiling - 1.0) * m
    recon = _recon(out["x_hat"], x, sigma)
    kl_per_dim = kl_per_dim_gaussian(out["mu"], out["logvar"], prior_mu)
    kl = kl_per_dim.sum()

    z_content, _ = model.split(out["z"])
    if adv_weight == 0:
        # Adversary is built but inert: no loss term and no gradient (diagnostics only).
        with torch.no_grad():
            logits = model.adversary_logits(z_content.detach(), 0.0)
            adv = F.binary_cross_entropy_with_logits(logits, smiling)
        total = recon + beta * kl
    else:
        logits = model.adversary_logits(z_content, lambd)
        adv = F.binary_cross_entropy_with_logits(logits, smiling)
        total = recon + beta * kl + adv_weight * adv
    adv_acc = ((logits.detach() > 0).to(smiling.dtype) == smiling).to(smiling.dtype).mean()
    # How well the supervised axis itself separates the classes (sign of its posterior mean).
    smile_acc = ((out["mu"][:, -1].detach() > 0).to(smiling.dtype) == smiling).to(smiling.dtype).mean()
    return {"loss": total, "recon": recon, "kl": kl, "kl_per_dim": kl_per_dim,
            "adv": adv, "adv_acc": adv_acc, "smile_acc": smile_acc}
