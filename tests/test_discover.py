import json

import pytest
import torch

from src.discover_smile import discover


@pytest.fixture(scope="module")
def discovered(baseline_run, synthetic_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    d = discover(baseline_run, data_root=synthetic_root, evaluator_path=None, out_dir=out, max_train=200)
    return d, out


def test_directions_file_structure(discovered, baseline_run):
    from pathlib import Path

    d, _ = discovered
    saved = torch.load(Path(baseline_run) / "directions.pt", weights_only=False)
    for key in ("mean_diff", "probe", "coordinate"):
        e = saved[key]
        assert set(["v", "p05", "p95", "val_select_acc"]) <= set(e)
        assert e["v"].shape == (16,)
        assert abs(float(e["v"].norm()) - 1.0) < 1e-5
        assert e["p05"] < e["p95"]
        assert 0.0 <= e["val_select_acc"] <= 1.0
    assert "index" in saved["coordinate"]
    assert saved["coordinate"]["v"].abs().max() == 1.0
    assert "label_efficiency" in saved
    # synthetic val_fit has only 45 rows, so budgets of 100 and above are skipped
    assert "100" not in saved["label_efficiency"]
    assert "all" in saved["label_efficiency"]
    for ent in saved["label_efficiency"].values():
        for k in ("mean_diff", "probe"):
            assert ent[k]["p05"] < ent[k]["p95"]
            assert abs(float(ent[k]["v"].norm()) - 1.0) < 1e-5
    # sizes above what is available are skipped (synthetic val_fit is small)
    assert "10000" not in saved["label_efficiency"]


def test_outputs_written(discovered):
    _, out = discovered
    jsons = list((out / "metrics").glob("*_coordinates.json"))
    assert len(jsons) == 1
    rep = json.loads(jsons[0].read_text())
    assert rep["method"] == "label_correlation"
    assert len(rep["response"]) == 16
    pngs = list((out / "latent_traversals").glob("*/*.png"))
    assert any(p.name == "direction_edits.png" for p in pngs)
    assert any(p.name.startswith("traversal_face") for p in pngs)


def test_supervised_rejected(supervised_run, synthetic_root, tmp_path):
    with pytest.raises(ValueError, match="only to plain VAEs"):
        discover(supervised_run, data_root=synthetic_root, out_dir=tmp_path)
