import json

import numpy as np
import pytest
import torch

from src import metrics as M
from src.evaluators import (AttributeEvaluator, IdentityEmbedder, load_attribute_evaluator, train_evaluator,
                            validate_evaluator)


def test_recon_mse_mae():
    x = np.zeros((2, 3, 4, 4))
    y = np.zeros((2, 3, 4, 4))
    y[1] = 0.5
    assert np.allclose(M.recon_mse(x, y), [0.0, 0.25])
    assert np.allclose(M.recon_mae(x, y), [0.0, 0.5])


def test_ssim():
    rng = np.random.default_rng(0)
    x = rng.random((2, 3, 32, 32))
    assert np.allclose(M.recon_ssim(x, x), 1.0)
    assert M.recon_ssim(x, rng.random((2, 3, 32, 32))).max() < 0.5


def test_monotonicity():
    s = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.1, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1], [0.1, 0.3, 0.2, 0.4, 0.5]])
    assert M.monotonic_mask(s).tolist() == [True, False, False, False]
    assert M.monotonic_fraction(s) == 0.25
    sp = M.spearman_per_image(s)
    assert np.isclose(sp[0], 1.0) and np.isclose(sp[2], -1.0)
    assert np.isclose(sp[1], 0.97467943, atol=1e-6)  # one tie: hand-computed
    assert np.isclose(sp[3], 0.9)
    assert np.allclose(M.score_range(s), [0.4, 0.4, -0.4, 0.4])
    assert M.spearman_per_image(np.full((1, 5), 0.3))[0] == 0.0
    m = M.monotonicity_metrics(s)
    assert set(m) == {"monotonic_fraction", "spearman_mean", "score_range_mean"}
    assert np.isclose(m["score_range_mean"], 0.2)


def test_cosine_and_groups():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    b = np.array([[1.0, 0.0], [-1.0, -1.0], [1.0, 0.0]])
    assert np.allclose(M.cosine_similarity(a, b), [1.0, -1.0, 0.0])
    names = ["Smiling", "Eyeglasses", "Rosy_Cheeks", "Male", "Other"]
    g = M.attribute_groups(names)
    assert g["unrelated"] == [1, 3] and g["smile_entailed"] == [2]
    assert not set(g["unrelated"]) & set(g["smile_entailed"])
    assert np.allclose(M.mean_abs_change(np.array([[0.0, 1.0, 0.5]]), np.array([[0.5, 0.0, 0.5]]), [0, 1]), [0.75])


def test_identity_embedder_shape_and_norm():
    emb = IdentityEmbedder(device=torch.device("cpu"), pretrained=False)
    out = emb.embed(torch.rand(3, 3, 64, 64) * 2 - 1)
    assert out.shape == (3, 512)
    assert torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-4)


def test_attribute_evaluator_shapes():
    ev = AttributeEvaluator([f"a{i}" for i in range(40)], 64, width=8)
    assert ev(torch.zeros(2, 3, 64, 64)).shape == (2, 40)
    assert ev(torch.zeros(2, 3, 128, 128)).shape == (2, 40)


def test_train_and_validate_evaluator(synthetic_root, baseline_run, tmp_path):
    path = train_evaluator(synthetic_root, tmp_path / "evaluator", recon_run=baseline_run, epochs=1,
                           batch_size=64, width=8, device=torch.device("cpu"))
    assert path.exists()
    ev = load_attribute_evaluator(path, torch.device("cpu"))
    assert len(ev.attr_names) == 40 and not ev.training
    assert ev.info["recon_run"] == str(baseline_run)
    rep = validate_evaluator(path, synthetic_root, baseline_run, tmp_path / "results", identity=None,
                             device=torch.device("cpu"))
    f = tmp_path / "results" / "metrics" / "evaluator_validation.json"
    assert f.exists()
    data = json.loads(f.read_text())
    assert data["split"] == "val_select"
    for k in ("real", "reconstruction"):
        assert 0.0 <= data["smile"][k]["accuracy"] <= 1.0
        assert len(data["attributes"][k]) == 40
    assert data["identity"]["skipped"] is True
    rel = data["reliability"]
    assert rel["thresholds"]["smile_auc_min"] == 0.9
    assert isinstance(rel["smile"], bool) and rel["identity"] is None
    assert len(rel["per_attribute"]) == 40
    assert rep["n_images"] == data["n_images"]


def test_train_without_recon_run(synthetic_root, tmp_path):
    path = train_evaluator(synthetic_root, tmp_path / "ev", recon_run=None, epochs=1, subset=128, batch_size=64,
                           width=8, device=torch.device("cpu"))
    info = load_attribute_evaluator(path, torch.device("cpu")).info
    assert info["recon_run"] is None and "blur" in info["augmentation"].lower()
