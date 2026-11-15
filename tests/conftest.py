import os

os.environ["LATENT_SMILE_DEVICE"] = "cpu"  # before any torch import

import copy  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    from src.data import make_synthetic

    root = tmp_path_factory.mktemp("synthetic") / "synthetic_64"
    make_synthetic(root, n=600, size=64, seed=0)
    return str(root)


@pytest.fixture(scope="session")
def tiny_cfg(synthetic_root, tmp_path_factory):
    """Factory: ``tiny_cfg("vae")`` / ``tiny_cfg("supervised")`` -> fresh complete config dict."""
    from src.config import fill_defaults

    out_dir = str(tmp_path_factory.mktemp("runs"))

    def make(model_type="vae"):
        assert model_type in ("vae", "supervised")
        cfg = fill_defaults({
            "name": f"tiny_{model_type}",
            "model": {"type": model_type, "image_size": 64, "latent_dim": 16, "base_channels": 8,
                      "smile_prior_mean": 2.0, "adv_hidden": 16},
            "loss": {"sigma": 1.0, "beta": 1.0, "kl_warmup_epochs": 1, "adv_weight": 1.0,
                     "adv_warmup_epochs": 1},
            "train": {"data_root": synthetic_root, "out_dir": out_dir, "epochs": 2, "batch_size": 64,
                      "lr": 0.001, "seed": 0, "subset": None, "num_workers": 0, "flip": True},
        })
        return copy.deepcopy(cfg)

    return make


@pytest.fixture(scope="session")
def baseline_run(tiny_cfg):
    from src.train import train

    return train(tiny_cfg("vae"))


@pytest.fixture(scope="session")
def supervised_run(tiny_cfg):
    from src.train import train

    return train(tiny_cfg("supervised"))
