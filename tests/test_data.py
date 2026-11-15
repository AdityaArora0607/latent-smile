import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ATTR_NAMES, FaceDataset, SPLITS, make_loader, make_synthetic, to_unit  # noqa: E402


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_celeba", ROOT / "scripts" / "prepare_celeba.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def syn_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn") / "synthetic_64"
    make_synthetic(root, n=200, size=64, seed=0)
    return root


def test_attr_names_are_official():
    assert len(ATTR_NAMES) == 40
    assert ATTR_NAMES[0] == "5_o_Clock_Shadow"
    assert ATTR_NAMES[-1] == "Young"
    assert ATTR_NAMES.index("Smiling") == 31


def test_shapes_dtypes_and_range(syn_root):
    ds = FaceDataset(syn_root, split="train")
    assert ds.image_size == 64
    assert ds.smile_index == ATTR_NAMES.index("Smiling")
    assert ds.attr_names == ATTR_NAMES
    x, y, idx = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert y.shape == (40,) and y.dtype == torch.float32
    assert set(np.unique(y.numpy())).issubset({0.0, 1.0})
    assert isinstance(idx, int)
    assert ds.attrs.dtype == np.int8 and ds.identity.dtype == np.int32

    loader = make_loader(ds, batch_size=16, shuffle=True)
    xb, yb, ib = next(iter(loader))
    assert xb.shape == (16, 3, 64, 64) and xb.dtype == torch.float32
    assert float(xb.min()) >= -1.0 and float(xb.max()) <= 1.0
    assert float(xb.min()) < 0.0 < float(xb.max())


def test_splits_disjoint_and_cover_all(syn_root):
    train = FaceDataset(syn_root, "train").indices
    val = FaceDataset(syn_root, "val").indices
    test = FaceDataset(syn_root, "test").indices
    everything = FaceDataset(syn_root, "all").indices
    assert len(np.intersect1d(train, val)) == 0
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(val, test)) == 0
    union = np.sort(np.concatenate([train, val, test]))
    assert np.array_equal(union, np.arange(len(everything)))
    assert np.array_equal(everything, np.arange(200))
    assert len(train) == 140 and len(val) == 30 and len(test) == 30


def test_val_fit_and_select_partition_val(syn_root):
    val = FaceDataset(syn_root, "val").indices
    fit = FaceDataset(syn_root, "val_fit").indices
    sel = FaceDataset(syn_root, "val_select").indices
    assert len(np.intersect1d(fit, sel)) == 0
    assert np.array_equal(np.sort(np.concatenate([fit, sel])), val)
    assert np.array_equal(fit, val[0::2]) and np.array_equal(sel, val[1::2])
    assert set(SPLITS) == {"train", "val", "val_fit", "val_select", "test", "all"}


def test_subset_keeps_first_rows(syn_root):
    full = FaceDataset(syn_root, "train")
    sub = FaceDataset(syn_root, "train", subset=5)
    assert len(sub) == 5
    assert np.array_equal(sub.indices, full.indices[:5])
    assert sub.attrs.shape == (5, 40)
    assert np.array_equal(sub[3][0].numpy(), full[3][0].numpy())


def test_deterministic_without_flip(syn_root):
    a = FaceDataset(syn_root, "train", flip=False)
    b = FaceDataset(syn_root, "train", flip=False, seed=123)
    for i in range(5):
        xa, ya, ia = a[i]
        xb, yb, ib = b[i]
        assert torch.equal(xa, xb) and torch.equal(ya, yb) and ia == ib


def test_flip_changes_some_items(syn_root):
    plain = FaceDataset(syn_root, "train", flip=False)
    flipped = FaceDataset(syn_root, "train", flip=True, seed=0)
    changed = sum(
        not torch.equal(plain[i][0], flipped[i][0]) for i in range(len(plain))
    )
    assert 0 < changed < len(plain)


def test_to_unit():
    x = torch.tensor([-1.5, -1.0, 0.0, 1.0, 1.5])
    assert torch.allclose(to_unit(x), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    assert np.allclose(to_unit(np.array([-1.0, 1.0])), [0.0, 1.0])


def test_synthetic_smile_signal_is_visible(syn_root):
    """Smiles curve up: mouth pixels at the centre sit lower than at the corners."""
    imgs = np.load(syn_root / "images.npy", mmap_mode="r")
    s = np.load(syn_root / "attrs.npy")[:, ATTR_NAMES.index("Smiling")]
    band = np.asarray(imgs[:, 36:52, 16:48])  # [N,16,32,3] around the mouth
    # mouth colour is (150, 50, 60); match it closely so backgrounds do not leak in
    mouth = (np.abs(band.astype(np.int16) - np.array([150, 50, 60])).sum(-1) < 60).astype(np.float32)
    rows = np.arange(16, dtype=np.float32)[None, :, None]

    def mean_row(part):
        cnt = part.sum(axis=(1, 2))
        return (part * rows).sum(axis=(1, 2)) / np.maximum(cnt, 1), cnt

    centre, c_cnt = mean_row(mouth[:, :, 12:20])
    sides, s_cnt = mean_row(np.concatenate([mouth[:, :, 3:7], mouth[:, :, 25:29]], axis=2))
    ok = (c_cnt > 0) & (s_cnt > 0)
    curve = (centre - sides)[ok]
    smiling = s[ok] == 1
    assert smiling.any() and (~smiling).any()
    assert curve[smiling].mean() > 1.0
    assert abs(curve[~smiling].mean()) < 0.5


def test_synthetic_metadata(syn_root):
    import json

    meta = json.loads((syn_root / "meta.json").read_text())
    assert meta == {"image_size": 64, "crop": 148, "source": "synthetic"}
    split = np.load(syn_root / "split.npy")
    assert set(np.unique(split).tolist()) == {0, 1, 2}


def test_make_synthetic_small_size_and_seed(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    make_synthetic(a, n=12, size=32, seed=7)
    make_synthetic(b, n=12, size=32, seed=7)
    ia = np.load(a / "images.npy")
    ib = np.load(b / "images.npy")
    assert ia.shape == (12, 32, 32, 3) and ia.dtype == np.uint8
    assert np.array_equal(ia, ib)
    assert np.array_equal(np.load(a / "attrs.npy"), np.load(b / "attrs.npy"))


def _write_fake_raw(raw: Path, n: int = 6):
    (raw / "img_align_celeba").mkdir(parents=True)
    rng = np.random.default_rng(0)
    names = [f"{i + 1:06d}.jpg" for i in range(n)]
    for name in names:
        arr = rng.integers(0, 255, size=(218, 178, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(raw / "img_align_celeba" / name, quality=95)

    attr_lines = [str(n), " ".join(ATTR_NAMES)]
    for name in names:
        vals = rng.choice([-1, 1], size=40)
        attr_lines.append(name + " " + " ".join(str(int(v)) for v in vals))
    (raw / "list_attr_celeba.txt").write_text("\n".join(attr_lines) + "\n")

    (raw / "identity_CelebA.txt").write_text(
        "\n".join(f"{name} {100 + i % 3}" for i, name in enumerate(names)) + "\n"
    )
    parts = [0, 0, 0, 1, 2, 2][:n]
    (raw / "list_eval_partition.txt").write_text(
        "\n".join(f"{name} {p}" for name, p in zip(names, parts)) + "\n"
    )
    return names


def test_prepare_celeba_conversion(tmp_path):
    prep = _load_prepare_module()
    raw = tmp_path / "raw" / "celeba"
    names = _write_fake_raw(raw, n=6)
    out = tmp_path / "celeba_64"

    n = prep.prepare(raw, out, size=64)
    assert n == 6
    imgs = np.load(out / "images.npy", mmap_mode="r")
    assert imgs.shape == (6, 64, 64, 3) and imgs.dtype == np.uint8
    attrs = np.load(out / "attrs.npy")
    assert attrs.shape == (6, 40) and attrs.dtype == np.int8
    assert set(np.unique(attrs).tolist()) == {0, 1}
    assert np.load(out / "identity.npy").dtype == np.int32
    assert np.load(out / "split.npy").tolist() == [0, 0, 0, 1, 2, 2]
    assert (out / "attr_names.json").exists()

    import json

    assert json.loads((out / "attr_names.json").read_text()) == ATTR_NAMES
    assert json.loads((out / "meta.json").read_text()) == {
        "image_size": 64, "crop": 148, "source": "celeba"
    }

    # attribute values: -1 -> 0, 1 -> 1, in file order
    rows = (raw / "list_attr_celeba.txt").read_text().splitlines()[2:]
    expected = np.array([[int(v) > 0 for v in r.split()[1:]] for r in rows], dtype=np.int8)
    assert np.array_equal(attrs, expected)

    # the dataset loads through src.data
    ds = FaceDataset(out, "train")
    x, y, idx = ds[0]
    assert x.shape == (3, 64, 64) and float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert names[0] == "000001.jpg"


def test_prepare_celeba_crop_geometry(tmp_path):
    """At size 148 the output must equal the centre crop (left=15, top=35) exactly."""
    prep = _load_prepare_module()
    raw = tmp_path / "raw" / "celeba"
    _write_fake_raw(raw, n=2)
    src = np.asarray(Image.open(raw / "img_align_celeba" / "000001.jpg").convert("RGB"))
    out = tmp_path / "celeba_148"
    prep.prepare(raw, out, size=148)
    got = np.load(out / "images.npy")[0]
    assert got.shape == (148, 148, 3)
    assert np.array_equal(got, src[35:183, 15:163])


def test_prepare_celeba_limit(tmp_path):
    prep = _load_prepare_module()
    raw = tmp_path / "raw" / "celeba"
    _write_fake_raw(raw, n=6)
    out = tmp_path / "limited"
    assert prep.prepare(raw, out, size=32, limit=3) == 3
    assert np.load(out / "images.npy", mmap_mode="r").shape == (3, 32, 32, 3)
    assert np.load(out / "split.npy").tolist() == [0, 0, 0]
