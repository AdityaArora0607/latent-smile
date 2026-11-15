"""Write a synthetic drawn-face dataset in the src.data on-disk format.

Usage (from the project root):

    python scripts/make_synthetic.py --out data/synthetic_64 --n 2000 --size 64 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import make_synthetic  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "data/synthetic_64"), help="output dataset folder")
    parser.add_argument("--n", type=int, default=2000, help="number of images")
    parser.add_argument("--size", type=int, default=64, help="image size")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(argv)

    make_synthetic(args.out, n=args.n, size=args.size, seed=args.seed)
    print(f"Wrote {args.n} synthetic {args.size}x{args.size} faces to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
