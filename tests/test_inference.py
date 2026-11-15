import shutil

import pytest
import torch

from src.discover_smile import discover
from src.inference import SmileEditor


@pytest.fixture(scope="module")
def discovered_run(baseline_run, synthetic_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    discover(baseline_run, data_root=synthetic_root, out_dir=out, max_train=200)
    return baseline_run


@pytest.fixture(scope="module")
def images(synthetic_root):
    from src.data import FaceDataset

    ds = FaceDataset(synthetic_root, split="val_select")
    return torch.stack([ds[i][0] for i in range(6)])


@pytest.fixture(params=["supervised", "discovered"])
def editor(request, supervised_run, discovered_run):
    run = supervised_run if request.param == "supervised" else discovered_run
    return SmileEditor.load(run, device=torch.device("cpu"))


def test_kinds(supervised_run, discovered_run):
    assert SmileEditor.load(supervised_run).kind == "supervised"
    for d in ("probe", "mean_diff", "coordinate"):
        ed = SmileEditor.load(discovered_run, direction=d)
        assert ed.kind == "discovered"
        assert ed.image_size == 64


def test_edit_levels_shape(editor, images):
    out = editor.edit_levels(images, 5)
    assert out.shape == (6, 5, 3, 64, 64)
    assert out.abs().max() <= 1.0 + 1e-6


def test_deterministic(editor, images):
    assert torch.equal(editor.edit_levels(images), editor.edit_levels(images))
    assert torch.equal(editor.encode(images), editor.encode(images))


def test_absolute_target(editor, images):
    for t in (0.0, 0.5, 1.0, 1.4):
        c = editor.smile_coordinate(editor.edit_at(images, t))
        # re-encoding is not exact, so check the latent-level property directly
        z = editor._edit_z(editor.encode(images), editor._targets(t, images.shape[0]))
        coord = z[:, -1] if editor.kind == "supervised" else z @ editor.v
        assert torch.allclose(coord, coord[0].expand_as(coord), atol=1e-4)
        lo, hi = editor.target_range
        assert torch.allclose(coord[0], torch.tensor(lo + t * (hi - lo)), atol=1e-4)
        assert c.shape == (6,)


def test_edit_at_tensor_t(editor, images):
    t = torch.linspace(0, 1, images.shape[0])
    out = editor.edit_at(images, t)
    assert out.shape == images.shape
    for i in (0, 3):
        single = editor.edit_at(images[i : i + 1], float(t[i]))
        assert torch.allclose(out[i : i + 1], single, atol=1e-5)
    with pytest.raises(ValueError):
        editor.edit_at(images, torch.zeros(3))


def test_smile_coordinate_monotone_in_levels(editor, images):
    z = editor.encode(images)
    coords = []
    for t in torch.linspace(0, 1, 5):
        zt = editor._edit_z(z, editor._targets(float(t), z.shape[0]))
        coords.append(zt[:, -1] if editor.kind == "supervised" else zt @ editor.v)
    coords = torch.stack(coords, 1)
    assert (coords[:, 1:] > coords[:, :-1]).all()


def test_content_code(editor, images):
    c = editor.content_code(images)
    D = editor.model.latent_dim
    if editor.kind == "supervised":
        assert c.shape == (6, D - 1)
    else:
        assert c.shape == (6, D)
        assert torch.allclose(c @ editor.v, torch.zeros(6), atol=1e-5)


def test_reconstruct_shape(editor, images):
    assert editor.reconstruct(images).shape == images.shape


def test_direction_entry_override(baseline_run, discovered_run):
    entry = torch.load(f"{baseline_run}/directions.pt", weights_only=False)["label_efficiency"]["all"]["probe"]
    ed = SmileEditor.load(baseline_run, direction_entry=entry)
    assert ed.kind == "discovered"
    assert torch.allclose(ed.v, entry["v"])


def test_missing_directions_error(baseline_run, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(baseline_run, run)
    (run / "directions.pt").unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError, match="python -m src.discover_smile --run"):
        SmileEditor.load(run)
