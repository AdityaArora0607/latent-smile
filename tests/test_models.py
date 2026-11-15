import math

import pytest
import torch

from src.checkpoint import load_model, save_checkpoint
from src.config import load_config
from src.losses import supervised_loss, vae_loss
from src.models import SupervisedVAE, VAE, build_model, grad_reverse


def _cfg(kind, size=64, z=16):
    return {"model": {"type": kind, "image_size": size, "latent_dim": z, "base_channels": 8,
                      "smile_prior_mean": 2.0, "adv_hidden": 8}}


@pytest.mark.parametrize("size", [64, 128])
@pytest.mark.parametrize("kind", ["vae", "supervised"])
def test_shapes(size, kind):
    model = build_model(_cfg(kind, size))
    assert isinstance(model, SupervisedVAE if kind == "supervised" else VAE)
    x = torch.rand(3, 3, size, size) * 2 - 1
    mu, logvar = model.encode(x)
    assert mu.shape == logvar.shape == (3, 16)
    out = model(x)
    assert out["x_hat"].shape == x.shape
    assert out["z"].shape == (3, 16)
    assert out["x_hat"].abs().max() <= 1.0


def test_block_count():
    from src.models.vae import num_blocks
    assert num_blocks(64) == 4 and num_blocks(128) == 5


def test_split_picks_last_dim():
    model = build_model(_cfg("supervised"))
    z = torch.arange(32.0).view(2, 16)
    content, smile = model.split(z)
    assert content.shape == (2, 15) and smile.shape == (2,)
    assert torch.equal(smile, z[:, -1]) and torch.equal(content, z[:, :-1])
    assert model.adversary_logits(content).shape == (2,)


def test_decoder_depends_on_z_smile():
    model = build_model(_cfg("supervised")).eval()
    z = torch.randn(2, 16)
    z2 = z.clone()
    z2[:, -1] += 4.0
    with torch.no_grad():
        assert (model.decode(z) - model.decode(z2)).abs().max() > 0


def test_vae_loss_hand_checked():
    out = {"x_hat": torch.zeros(2, 3, 4, 4), "mu": torch.zeros(2, 5), "logvar": torch.zeros(2, 5)}
    x = torch.full((2, 3, 4, 4), 0.5)
    r = vae_loss(out, x, sigma=1.0, beta=2.0)
    assert r["recon"].item() == pytest.approx(48 * 0.25 / 2)
    assert r["kl"].item() == pytest.approx(0.0)
    out["mu"] = torch.ones(2, 5)
    r = vae_loss(out, x, sigma=0.5, beta=2.0)
    assert r["recon"].item() == pytest.approx(48 * 0.25 / (2 * 0.25))
    assert r["kl_per_dim"].shape == (5,)
    assert r["kl"].item() == pytest.approx(5 * 0.5)  # 0.5 * mu^2 per dim
    assert r["loss"].item() == pytest.approx(r["recon"].item() + 2.0 * 2.5)


def test_class_conditional_kl():
    model = build_model(_cfg("supervised", z=4))
    m = model.smile_prior_mean
    x = torch.zeros(2, 3, 64, 64)
    mu = torch.zeros(2, 4)
    mu[:, -1] = torch.tensor([m, -m])  # sits exactly on the prior mean of its own class
    out = {"x_hat": x.clone(), "mu": mu, "logvar": torch.zeros(2, 4), "z": mu.clone()}
    smiling = torch.tensor([1.0, 0.0])
    r = supervised_loss(model, out, x, smiling, 1.0, 1.0, 1.0, 1.0)
    assert r["kl"].item() == pytest.approx(0.0, abs=1e-6)
    # swap labels: each is 2m away -> 0.5*(2m)^2 per sample
    r = supervised_loss(model, out, x, 1 - smiling, 1.0, 1.0, 1.0, 1.0)
    assert r["kl"].item() == pytest.approx(0.5 * (2 * m) ** 2)
    assert torch.isfinite(r["adv"]) and 0 <= r["adv_acc"].item() <= 1
    assert r["kl_per_dim"][:-1].abs().max() == 0


def test_gradient_reversal_sign():
    x = torch.ones(3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))
    x = torch.ones(3, requires_grad=True)
    (grad_reverse(x, 1.0) * 2).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -2.0))


def _encoder_grad_norm(model, x, smiling, adv_weight, beta=0.0, sigma=1e6):
    model.zero_grad()
    out = model(x)
    # sigma huge and beta 0: only the adversary term can reach the encoder
    r = supervised_loss(model, out, x, smiling, sigma, beta, adv_weight, 1.0)
    r["loss"].backward()
    return sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)


def test_adv_weight_zero_gives_no_adversary_gradient():
    torch.manual_seed(0)
    model = build_model(_cfg("supervised"))
    x = torch.rand(8, 3, 64, 64) * 2 - 1
    smiling = (torch.rand(8) > 0.5).float()
    # Isolate the adversary: zero out the recon/KL contributions via beta=0 and huge sigma.
    g_off = _encoder_grad_norm(model, x, smiling, adv_weight=0.0)
    assert g_off < 1e-6
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.adversary.parameters())
    g_on = _encoder_grad_norm(model, x, smiling, adv_weight=1.0)
    assert g_on > 1e-3


def test_checkpoint_round_trip(tmp_path):
    cfg = load_config("configs/baseline.yaml", {"model.latent_dim": 16, "model.base_channels": 8})
    cfg["model"]["type"] = "supervised"
    model = build_model(cfg).eval()
    opt = torch.optim.Adam(model.parameters())
    save_checkpoint(tmp_path / "last.pt", model, opt, 3, cfg, best_val=1.5)
    loaded, cfg2 = load_model(tmp_path)  # directory falls back to last.pt
    assert cfg2 == cfg and isinstance(loaded, SupervisedVAE) and not loaded.training
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        assert torch.equal(model.decode(model.encode(x)[0]), loaded.decode(loaded.encode(x)[0]))


def test_trained_checkpoint_loads(supervised_run):
    model, cfg = load_model(supervised_run)
    assert isinstance(model, SupervisedVAE) and cfg["model"]["type"] == "supervised"
    assert math.isfinite(model.smile_prior_mean)
