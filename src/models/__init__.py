from .supervised_vae import GradReverse, SupervisedVAE, grad_reverse
from .vae import VAE


def build_model(cfg: dict) -> VAE:
    """Build a model from a complete config dict (dispatches on ``model.type``)."""
    m = cfg["model"]
    kind = m.get("type", "vae")
    common = dict(image_size=m["image_size"], latent_dim=m["latent_dim"],
                  base_channels=m["base_channels"])
    if kind == "vae":
        return VAE(**common)
    if kind == "supervised":
        return SupervisedVAE(**common, smile_prior_mean=m["smile_prior_mean"],
                             adv_hidden=m["adv_hidden"])
    raise ValueError(f"unknown model type: {kind!r}")


__all__ = ["VAE", "SupervisedVAE", "GradReverse", "grad_reverse", "build_model"]
