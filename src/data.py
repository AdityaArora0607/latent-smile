"""Dataset access for the prepared CelebA / synthetic face datasets.

On-disk format (see the shared contract):

    images.npy          uint8 [N,S,S,3] RGB, memory-mapped on read
    attrs.npy           int8  [N,40] in {0,1}
    attr_names.json     list of the 40 CelebA attribute names, file order
    identity.npy        int32 [N]
    split.npy           int8  [N]: 0 train / 1 val / 2 test
    meta.json           {"image_size": S, "crop": 148, "source": "celeba" | "synthetic"}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "val", "val_fit", "val_select", "test", "all")

ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
    "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
    "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]
assert len(ATTR_NAMES) == 40

_SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def _load_meta(root: Path) -> dict:
    with open(root / "meta.json") as f:
        return json.load(f)


class FaceDataset(Dataset):
    def __init__(self, root, split="train", subset=None, flip=False, seed=0):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.flip = flip
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        meta = _load_meta(self.root)
        self.image_size = int(meta["image_size"])
        with open(self.root / "attr_names.json") as f:
            self.attr_names = list(json.load(f))
        self.smile_index = self.attr_names.index("Smiling")

        all_attrs = np.load(self.root / "attrs.npy")
        all_ids = np.load(self.root / "identity.npy")
        all_split = np.load(self.root / "split.npy")
        n = len(all_split)

        if split == "all":
            idx = np.arange(n, dtype=np.int64)
        elif split in _SPLIT_CODE:
            idx = np.nonzero(all_split == _SPLIT_CODE[split])[0].astype(np.int64)
        else:  # val_fit / val_select: even / odd positions of the val rows
            val_idx = np.nonzero(all_split == _SPLIT_CODE["val"])[0].astype(np.int64)
            idx = val_idx[0::2] if split == "val_fit" else val_idx[1::2]

        if subset is not None:
            idx = idx[: int(subset)]

        self.indices = idx
        self.attrs = np.ascontiguousarray(all_attrs[idx]).astype(np.int8)
        self.identity = np.ascontiguousarray(all_ids[idx]).astype(np.int32)
        self._images = None  # opened lazily so the dataset pickles cleanly for workers
        self._image_path = self.root / "images.npy"

    def __len__(self) -> int:
        return len(self.indices)

    def _image_array(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self._image_path, mmap_mode="r")
        return self._images

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        img = np.array(self._image_array()[row], dtype=np.uint8)  # [S,S,3]
        x = torch.from_numpy(img).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        if self.flip and self._rng.random() < 0.5:
            x = torch.flip(x, dims=[2])
        y = torch.from_numpy(self.attrs[i].astype(np.float32))
        return x, y, row


def make_loader(dataset, batch_size, shuffle=False, num_workers=0, drop_last=False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def to_unit(x):
    """Map [-1,1] to [0,1] and clamp."""
    if isinstance(x, torch.Tensor):
        return ((x + 1) / 2).clamp(0.0, 1.0)
    return np.clip((np.asarray(x) + 1) / 2, 0.0, 1.0)


def write_metadata(root, attrs, identity, split, image_size, source, crop=148) -> None:
    """Write everything except images.npy (which the caller streams / writes)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "attrs.npy", np.asarray(attrs, dtype=np.int8))
    np.save(root / "identity.npy", np.asarray(identity, dtype=np.int32))
    np.save(root / "split.npy", np.asarray(split, dtype=np.int8))
    with open(root / "attr_names.json", "w") as f:
        json.dump(ATTR_NAMES, f)
    with open(root / "meta.json", "w") as f:
        json.dump({"image_size": int(image_size), "crop": int(crop), "source": source}, f)


# ---------------------------------------------------------------------------
# Synthetic faces
# ---------------------------------------------------------------------------

_HAIR_COLOURS = {
    "black": (25, 20, 20),
    "blond": (225, 190, 110),
    "brown": (110, 70, 40),
    "gray": (180, 180, 180),
}
_MOUTH = (150, 50, 60)
_DARK = (40, 30, 30)


def _draw_face(S, skin, hair, bg, cx, cy, face_w, face_h, smile, glasses):
    """Draw one simple face. Smile = upward-curved mouth; no smile = flat mouth."""
    img = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(img)
    rx, ry = face_w * S, face_h * S
    shade = tuple(int(c * 0.8) for c in skin)

    if hair is not None:  # hair cap behind the head
        d.ellipse([cx - rx * 1.1, cy - ry * 1.15, cx + rx * 1.1, cy - ry * 0.1], fill=hair)
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=skin)
    if hair is not None:  # fringe over the forehead
        d.chord([cx - rx * 1.02, cy - ry * 1.02, cx + rx * 1.02, cy - ry * 0.25], 180, 360, fill=hair)

    ex, ey, er = rx * 0.42, cy - ry * 0.18, S * 0.035
    brow = hair if hair is not None else _DARK
    lw = max(1, S // 50)
    for sx in (-1, 1):
        x = cx + sx * ex
        d.ellipse([x - er * 1.4, ey - er, x + er * 1.4, ey + er], fill=(245, 245, 245))
        d.ellipse([x - er * 0.7, ey - er * 0.8, x + er * 0.7, ey + er * 0.8], fill=_DARK)
        d.line([x - er * 1.8, ey - er * 2.3, x + er * 1.8, ey - er * 2.3], fill=brow, width=lw)
    if glasses:
        for sx in (-1, 1):
            x = cx + sx * ex
            d.rectangle([x - er * 2.2, ey - er * 1.6, x + er * 2.2, ey + er * 1.6],
                        outline=(30, 30, 30), width=lw)
        d.line([cx - ex + er * 2.2, ey, cx + ex - er * 2.2, ey], fill=(30, 30, 30), width=lw)

    d.line([cx, ey + er, cx - S * 0.012, cy + ry * 0.25], fill=shade, width=max(1, S // 64))

    my = cy + ry * 0.5
    mw = rx * 0.45
    width = max(2, S // 16)
    if smile:
        mh = ry * 0.25
        d.arc([cx - mw, my - mh, cx + mw, my + mh], 0, 180, fill=_MOUTH, width=width)
    else:
        d.line([cx - mw, my, cx + mw, my], fill=_MOUTH, width=width)
    return np.asarray(img, dtype=np.uint8)


def make_synthetic(root, n=600, size=64, seed=0) -> None:
    """Write a synthetic face dataset in the on-disk format.

    Identities (~n/4) fix skin tone, hair colour, face shape and sex. Each image adds
    position jitter, background colour, and random Smiling / Eyeglasses. Smiling=1 draws
    an upward-curved mouth, 0 a flat one, so the smile is visually learnable.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = int(n)
    S = int(size)
    n_ids = max(1, n // 4)

    # identity-level parameters
    t = rng.uniform(0.3, 0.95, size=n_ids)
    skins = np.stack([
        np.clip(230 * t + 20, 0, 255), np.clip(180 * t + 15, 0, 255), np.clip(150 * t + 10, 0, 255)
    ], axis=1).astype(int)
    hair_names = np.array(["black", "blond", "brown", "gray", "bald"])
    hair_choice = hair_names[rng.integers(0, len(hair_names), size=n_ids)]
    male_id = rng.random(n_ids) < 0.5
    face_w = rng.uniform(0.27, 0.36, size=n_ids)
    face_h = rng.uniform(0.34, 0.42, size=n_ids)

    # per-image parameters
    ids = np.arange(n) % n_ids
    rng.shuffle(ids)
    smiling = (rng.random(n) < 0.5).astype(np.int8)
    glasses = (rng.random(n) < 0.25).astype(np.int8)
    jit = rng.normal(0, 0.03, size=(n, 2))
    bgs = rng.uniform(40, 220, size=(n, 3)).astype(int)

    attrs = (rng.random((n, 40)) < 0.3).astype(np.int8)
    col = {name: i for i, name in enumerate(ATTR_NAMES)}
    hair_per_row = hair_choice[ids]
    attrs[:, col["Smiling"]] = smiling
    attrs[:, col["Eyeglasses"]] = glasses
    attrs[:, col["Male"]] = male_id[ids].astype(np.int8)
    attrs[:, col["Bald"]] = (hair_per_row == "bald").astype(np.int8)
    for hname, aname in (("black", "Black_Hair"), ("blond", "Blond_Hair"),
                         ("brown", "Brown_Hair"), ("gray", "Gray_Hair")):
        attrs[:, col[aname]] = (hair_per_row == hname).astype(np.int8)

    images_path = root / "images.npy"
    from numpy.lib.format import open_memmap
    images = open_memmap(images_path, mode="w+", dtype=np.uint8, shape=(n, S, S, 3))
    for i in range(n):
        k = ids[i]
        h = hair_per_row[i]
        hair = None if h == "bald" else _HAIR_COLOURS[h]
        images[i] = _draw_face(
            S,
            tuple(int(c) for c in skins[k]),
            hair,
            tuple(int(c) for c in bgs[i]),
            S * 0.5 + jit[i, 0] * S,
            S * 0.5 + jit[i, 1] * S,
            face_w[k],
            face_h[k],
            bool(smiling[i]),
            bool(glasses[i]),
        )
    images.flush()
    del images

    # official-style 70/15/15 split over rows
    n_tr = int(round(0.70 * n))
    n_va = int(round(0.15 * n))
    perm = rng.permutation(n)
    split = np.full(n, 2, dtype=np.int8)
    split[perm[:n_tr]] = 0
    split[perm[n_tr:n_tr + n_va]] = 1

    write_metadata(root, attrs, ids.astype(np.int32), split, S, source="synthetic")
