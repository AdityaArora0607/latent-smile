"""Convert a raw CelebA folder into the on-disk dataset format used by src.data.

Usage (from the project root):

    python scripts/prepare_celeba.py --raw data/raw/celeba --out data/celeba_64 --size 64
    python scripts/prepare_celeba.py --download            # try torchvision download first

Expected raw layout (standard CelebA file names):

    img_align_celeba/         202,599 aligned 178x218 jpgs
    list_attr_celeba.txt      count line, header line of 40 names, rows "filename -1/1 x40"
    identity_CelebA.txt       rows "filename id"
    list_eval_partition.txt   rows "filename 0|1|2" (train / val / test)

Preprocessing: centre crop 148x148 (left=15, top=35), then Lanczos resize to SxS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ATTR_NAMES, write_metadata  # noqa: E402

CROP = 148
CROP_LEFT, CROP_TOP = 15, 35
REQUIRED = (
    "img_align_celeba",
    "list_attr_celeba.txt",
    "identity_CelebA.txt",
    "list_eval_partition.txt",
)

MANUAL_DOWNLOAD = """
Manual setup (the Google Drive source is often quota-blocked):
  1. Download the aligned images zip (img_align_celeba.zip) and the annotation files
     list_attr_celeba.txt, identity_CelebA.txt and list_eval_partition.txt from the
     official CelebA page (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or a mirror.
  2. Unzip so that the folder looks like this:
       data/raw/celeba/img_align_celeba/000001.jpg ...
       data/raw/celeba/list_attr_celeba.txt
       data/raw/celeba/identity_CelebA.txt
       data/raw/celeba/list_eval_partition.txt
  3. Re-run: python scripts/prepare_celeba.py --raw data/raw/celeba --out data/celeba_64 --size 64
"""


def _read_pairs(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = int(parts[1])
    return out


def _read_attr_file(path: Path) -> tuple[list[str], list[tuple[str, list[int]]]]:
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines()]
    n = int(lines[0].strip())
    names = lines[1].split()
    if len(names) != 40 or names != ATTR_NAMES:
        raise ValueError(f"{path} header does not match the 40 CelebA attribute names")
    rows = []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        rows.append((parts[0], [int(v) for v in parts[1:41]]))
    if len(rows) != n:
        raise ValueError(f"{path} declares {n} rows but has {len(rows)}")
    return names, rows


def _load_crop(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    crop = img.crop((CROP_LEFT, CROP_TOP, CROP_LEFT + CROP, CROP_TOP + CROP))
    if size != CROP:
        crop = crop.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(crop, dtype=np.uint8)


def prepare(raw, out, size, limit=None) -> int:
    """Convert a raw CelebA folder into the dataset format. Returns the number of rows."""
    raw, out = Path(raw), Path(out)
    img_dir = raw / "img_align_celeba"
    attr_names, rows = _read_attr_file(raw / "list_attr_celeba.txt")
    identities = _read_pairs(raw / "identity_CelebA.txt")
    partition = _read_pairs(raw / "list_eval_partition.txt")
    if limit is not None:
        rows = rows[: int(limit)]
    n = len(rows)
    size = int(size)

    out.mkdir(parents=True, exist_ok=True)
    images = open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    attrs = np.zeros((n, 40), dtype=np.int8)
    identity = np.zeros(n, dtype=np.int32)
    split = np.zeros(n, dtype=np.int8)

    for i, (fname, vals) in enumerate(tqdm(rows, desc=f"CelebA -> {size}px", unit="img")):
        path = img_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing image {path}")
        if fname not in identities:
            raise KeyError(f"{fname} not in identity file")
        if fname not in partition:
            raise KeyError(f"{fname} not in partition file")
        images[i] = _load_crop(path, size)
        attrs[i] = np.asarray(vals) > 0
        identity[i] = identities[fname]
        split[i] = partition[fname]

    images.flush()
    del images
    write_metadata(out, attrs, identity, split, size, source="celeba", crop=CROP)
    return n


def _have_raw(raw: Path) -> bool:
    return all((raw / name).exists() for name in REQUIRED)


def try_download(raw: Path) -> bool:
    """torchvision lays files out under <root>/celeba/, so root is the parent of raw."""
    try:
        from torchvision.datasets import CelebA

        CelebA(root=str(raw.parent), split="all", download=True)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure falls back to manual steps
        print(f"torchvision CelebA download failed: {exc}", file=sys.stderr)
        print(MANUAL_DOWNLOAD, file=sys.stderr)
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(ROOT / "data/raw/celeba"), help="raw CelebA folder")
    parser.add_argument("--out", default=str(ROOT / "data/celeba_64"), help="output dataset folder")
    parser.add_argument("--size", type=int, default=64, help="output image size (e.g. 64 or 128)")
    parser.add_argument("--limit", type=int, default=None, help="only convert the first N rows (debugging)")
    parser.add_argument("--download", action="store_true",
                        help="try torchvision.datasets.CelebA download into data/raw first")
    args = parser.parse_args(argv)

    raw = Path(args.raw)
    if args.download and not _have_raw(raw):
        if not try_download(raw):
            return 1

    missing = [name for name in REQUIRED if not (raw / name).exists()]
    if missing:
        print(f"Raw CelebA folder {raw} is missing: {', '.join(missing)}", file=sys.stderr)
        print(MANUAL_DOWNLOAD, file=sys.stderr)
        return 1

    n = prepare(raw, args.out, args.size, limit=args.limit)
    print(f"Wrote {n} rows to {args.out} ({args.size}x{args.size})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
