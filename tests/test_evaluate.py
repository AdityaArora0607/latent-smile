import csv
import json
from pathlib import Path

import pytest
import torch

from src.evaluate import FIELDS, SWEEP_T, aggregate, evaluate_run
from src.evaluators import AttributeEvaluator, IdentityEmbedder

CPU = torch.device("cpu")
N_IMG = 8  # the random-weight Inception embedder costs ~17 embeddings per image on CPU


@pytest.fixture(scope="module")
def evaluator(synthetic_root):
    from src.data import FaceDataset
    names = FaceDataset(synthetic_root, "test", subset=4).attr_names
    torch.manual_seed(0)
    return AttributeEvaluator(names, 64, width=8).eval()


@pytest.fixture(scope="module")
def identity():
    return IdentityEmbedder(CPU, pretrained=False)


def _check(res, out):
    assert res["n_images"] == N_IMG and res["split"] == "test"
    for k in ("mse", "mae"):
        assert res["reconstruction"][k] >= 0
    assert -1 <= res["reconstruction"]["ssim"] <= 1
    sc = res["smile_control"]
    assert len(sc["prob_mean"]) == 5 and all(0 <= p <= 1 for p in sc["prob_mean"])
    assert 0 <= sc["monotonicity"]["monotonic_fraction"] <= 1
    assert -1 <= sc["monotonicity"]["spearman_mean"] <= 1
    ide = res["identity"]
    assert ide["skipped"] is False
    for k in ("to_original", "to_reconstruction"):
        assert len(ide[k]["mean"]) == 5 and all(-1 <= v <= 1 for v in ide[k]["mean"])
    assert "reference_real_images" in ide
    lk = res["leakage"]
    assert "Rosy_Cheeks" in lk["smile_entailed"]["attributes"] and "Eyeglasses" in lk["unrelated"]["attributes"]
    assert 0 <= lk["unrelated"]["level1_to_level5"] <= 1 and len(lk["unrelated"]["level_vs_reconstruction"]) == 5
    tr = res["tradeoff"]
    assert tr["t"] == pytest.approx(SWEEP_T) and len(tr["t"]) == 9
    for k in ("smile_gain", "identity_sim_to_reconstruction", "leakage_unrelated"):
        assert len(tr[k]) == 9
    d = res["disentanglement"]
    for part in ("content_code", "full_latent"):
        assert 0 <= d[part]["linear_acc"] <= 1 and 0 <= d[part]["mlp_acc"] <= 1
    fc = res["failure_cases"]
    for k in ("lowest_smile_gain", "lowest_identity_similarity", "first_images"):
        assert Path(fc[k]["grid"]).exists()
    assert Path(res["files"]["reconstructions"]).exists()
    assert Path(res["files"]["tradeoff_figure"]).exists()
    tag = f"{res['name']}_seed{res['seed']}"
    assert (out / "metrics" / f"{tag}.json").exists()
    assert json.loads((out / "metrics" / f"{tag}.json").read_text())["name"] == res["name"]


def test_evaluate_supervised(supervised_run, evaluator, identity, synthetic_root, tmp_path):
    res = evaluate_run(supervised_run, evaluator, identity, data_root=synthetic_root, out_dir=tmp_path,
                       max_images=N_IMG, device=CPU)
    assert res["kind"] == "supervised" and res["label_efficiency"]["available"] is False
    _check(res, tmp_path)


def test_evaluate_discovered(baseline_run, evaluator, identity, synthetic_root, tmp_path):
    from src.discover_smile import discover
    discover(baseline_run, data_root=synthetic_root, out_dir=tmp_path / "disc")
    res = evaluate_run(baseline_run, evaluator, identity, data_root=synthetic_root, out_dir=tmp_path,
                       max_images=N_IMG, device=CPU)
    assert res["kind"] == "discovered"
    _check(res, tmp_path)
    le = res["label_efficiency"]
    assert le["available"] is True and len(le["budgets"]) >= 1
    for b in le["budgets"].values():
        assert len(b["prob_mean"]) == 5


def test_evaluate_no_identity(supervised_run, evaluator, synthetic_root, tmp_path):
    res = evaluate_run(supervised_run, evaluator, None, data_root=synthetic_root, out_dir=tmp_path,
                       max_images=32, device=CPU)
    assert res["identity"]["skipped"] is True
    assert res["tradeoff"]["identity_sim_to_reconstruction"] == {"skipped": True}
    assert res["failure_cases"]["lowest_identity_similarity"] == {"skipped": True}


def _fake(name, seed, gain_scale, with_identity=True):
    tr = {"t": SWEEP_T, "smile_gain": [gain_scale * t for t in SWEEP_T],
          "identity_sim_to_reconstruction": [0.9 - 0.1 * abs(t) for t in SWEEP_T] if with_identity else {"skipped": True},
          "leakage_unrelated": [0.01 * abs(t) for t in SWEEP_T]}
    return {
        "name": name, "seed": seed,
        "reconstruction": {"mse": 0.01 * (seed + 1), "mae": 0.05, "ssim": 0.8},
        "smile_control": {"prob_mean": [0.1, 0.3, 0.5, 0.7, 0.9 - seed * 0.1],
                          "monotonicity": {"monotonic_fraction": 0.5, "spearman_mean": 0.8, "score_range_mean": 0.7}},
        "identity": {"skipped": False, "to_original": {"mean": [0.9] * 5}, "to_reconstruction": {"mean": [0.95] * 5}}
        if with_identity else {"skipped": True},
        "leakage": {"unrelated": {"level1_to_level5": 0.02, "level_vs_reconstruction": [0.01] * 5},
                    "smile_entailed": {"level1_to_level5": 0.3, "level_vs_reconstruction": [0.1] * 5}},
        "tradeoff": tr,
    }


def test_aggregate(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    (m / "a_seed0.json").write_text(json.dumps(_fake("A", 0, 1.0)))
    (m / "a_seed1.json").write_text(json.dumps(_fake("A", 1, 1.0)))
    (m / "b_seed0.json").write_text(json.dumps(_fake("B", 0, 0.5, with_identity=False)))
    (m / "evaluator_validation.json").write_text(json.dumps({"reliability": {
        "smile": True, "unrelated_attributes": False, "smile_entailed_attributes": True, "identity": None,
        "thresholds": {"smile_auc_min": 0.9}}}))
    out = aggregate(tmp_path)
    rows = {r["model"]: r for r in out["rows"]}
    assert rows["A"]["n_seeds"] == 2 and rows["B"]["n_seeds"] == 1
    mean, std, n = rows["A"]["recon_mse"]
    assert mean == pytest.approx(0.015) and std == pytest.approx(0.00707107, abs=1e-6) and n == 2
    assert rows["B"]["recon_mse"][1] is None
    assert rows["B"]["identity_to_original_level1"][0] is None  # skipped, not zero
    with open(m / "summary.csv") as f:
        csv_rows = list(csv.DictReader(f))
    assert {r["model"] for r in csv_rows} == {"A", "B"}
    assert [r for r in csv_rows if r["model"] == "B"][0]["identity_to_original_level1_mean"] == ""
    md = (m / "summary.md").read_text()
    assert "†" in md and "unrelated_attributes" in md and "n/a" in md
    assert (tmp_path / "figures" / "tradeoff.png").exists()
    assert (tmp_path / "figures" / "smile_by_level.png").exists()
    assert len(FIELDS) == len({f[0] for f in FIELDS})
