import csv

import pytest
import yaml

from src.config import load_config
from src.train import ramp, train

EXPECTED = ["config.yaml", "last.pt", "best.pt", "log.csv", "curves.png", "recon.png"]


def _rows(run):
    with open(run / "log.csv") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("kind", ["vae", "supervised"])
def test_run_writes_files(kind, baseline_run, supervised_run):
    run = baseline_run if kind == "vae" else supervised_run
    for name in EXPECTED:
        assert (run / name).exists(), name
    rows = _rows(run)
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    assert all(float(r["val_recon"]) > 0 and float(r["val_kl"]) >= 0 for r in rows)
    assert "active_dims" in rows[0]
    if kind == "supervised":
        for col in ("val_adv", "val_adv_acc", "smile_sensitivity"):
            assert col in rows[0]
    else:
        assert "smile_sensitivity" not in rows[0]
    cfg = load_config(run / "config.yaml")
    assert cfg["model"]["type"] == kind


def test_resume_continues_epoch_count(tiny_cfg, tmp_path):
    cfg = tiny_cfg("supervised")
    cfg["train"].update(out_dir=str(tmp_path), epochs=1, subset=128)
    run = train(cfg)
    assert len(_rows(run)) == 1
    cfg["train"]["epochs"] = 3
    run2 = train(cfg, resume=True)
    assert run2 == run
    assert [int(r["epoch"]) for r in _rows(run)] == [1, 2, 3]


def test_vae_training_never_reads_labels(tiny_cfg, tmp_path, monkeypatch):
    """Corrupt the attrs tensor; a plain VAE run must be unaffected (never touches labels)."""
    import torch

    from src import train as T

    cfg = tiny_cfg("vae")
    cfg["train"].update(out_dir=str(tmp_path), epochs=1, subset=128)
    seen = {}
    orig = T.compute_loss

    def spy(model, batch, *a, **k):
        batch = (batch[0], torch.full_like(batch[1], float("nan")), batch[2])
        seen["ok"] = True
        return orig(model, batch, *a, **k)

    monkeypatch.setattr(T, "compute_loss", spy)
    run = train(cfg)
    assert seen["ok"] and (run / "best.pt").exists()
    assert all(float(r["train_loss"]) == float(r["train_loss"]) for r in _rows(run))  # not NaN


def test_warmup_ramp():
    assert ramp(0, 5) == 0 and ramp(2.5, 5) == 0.5 and ramp(9, 5) == 1 and ramp(0, 0) == 1


def test_config_overrides(tmp_path):
    cfg = load_config("configs/supervised.yaml", {"train.epochs": 2, "loss.adv_weight": 0.0})
    assert cfg["train"]["epochs"] == 2 and cfg["loss"]["adv_weight"] == 0.0
    assert cfg["model"]["type"] == "supervised"
    with pytest.raises(KeyError):
        load_config("configs/supervised.yaml", {"train.nope": 1})
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "model": {"latent_dim": 8}}))
    c = load_config(p)
    assert c["model"]["latent_dim"] == 8 and c["train"]["batch_size"] == 128 and c["name"] == "x"


@pytest.mark.parametrize("name", ["baseline", "supervised", "supervised_noadv", "beta_vae", "baseline_z32",
                                  "baseline_z128", "supervised_z32", "supervised_z128", "evaluator_vae",
                                  "demo128_baseline", "demo128_supervised"])
def test_shipped_configs_load(name):
    from src.models import build_model

    cfg = load_config(f"configs/{name}.yaml")
    assert cfg["name"] == name
    build_model(cfg)
