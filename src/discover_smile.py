"""Discover a smile direction in a plain (label-free) VAE.

Directions are fitted on ``val_fit`` labels only, scored on ``val_select``, and the edit
range (p05/p95) is calibrated on train-split projections without labels. The test split is
never touched.

    python -m src.discover_smile --run runs/baseline/seed0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import load_model
from src.data import FaceDataset, make_loader
from src.inference import SmileEditor
from src.utils import get_device

LABEL_SIZES = (100, 1000, 10000, "all")
N_TRAVERSAL_VALUES = 7
TRAVERSAL_RANGE = 3.0
MAX_TRAVERSAL_COORDS = 32


# ---------------------------------------------------------------------- encoding
@torch.no_grad()
def encode_split(model, dataset, device, batch_size: int = 256):
    """Posterior means [N,D] (float64 numpy) and smile labels [N] (int)."""
    loader = make_loader(dataset, batch_size, shuffle=False)
    zs = []
    for x, _, _ in loader:
        mu, _ = model.encode(x.to(device))
        zs.append(mu.cpu())
    Z = torch.cat(zs).double().numpy()
    y = np.asarray(dataset.attrs[:, dataset.smile_index]).astype(int)
    return Z, y


# ---------------------------------------------------------------------- fitting
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _threshold(Z, y, v):
    p = Z @ v
    return 0.5 * (p[y == 1].mean() + p[y == 0].mean())


def fit_mean_diff(Z, y):
    return _unit(Z[y == 1].mean(0) - Z[y == 0].mean(0))


def fit_probe(Z, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Z)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Z), y)
    scale = np.where(sc.scale_ > 1e-12, sc.scale_, 1.0)
    w = clf.coef_[0] / scale  # logit = sum_i w_i * z_i + const in the original latent space
    return _unit(w)


def fit_coordinate(Z, y):
    """Best single axis by absolute point-biserial correlation; sign-corrected one-hot."""
    corr = coordinate_correlations(Z, y)
    i = int(np.argmax(np.abs(corr)))
    v = np.zeros(Z.shape[1])
    v[i] = 1.0 if corr[i] >= 0 else -1.0
    return v, i


def coordinate_correlations(Z, y):
    yc = y - y.mean()
    Zc = Z - Z.mean(0)
    den = np.sqrt((Zc ** 2).sum(0) * (yc ** 2).sum())
    return np.where(den > 0, (Zc * yc[:, None]).sum(0) / np.where(den > 0, den, 1.0), 0.0)


def balanced_first(y, n):
    """Deterministic class-balanced selection of n indices (first of each class)."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    k = n // 2
    a, b = pos[:k], neg[: n - k]
    if len(a) < k:
        b = neg[: n - len(a)]
    if len(b) < n - k:
        a = pos[: n - len(b)]
    return np.sort(np.concatenate([a, b]))


def _entry(v, Ztr, Zf, yf, Zs, ys):
    """Direction entry: unit v, p05/p95 on unlabeled train projections, accuracy on val_select."""
    thr = _threshold(Zf, yf, v)
    acc = float(((Zs @ v > thr).astype(int) == ys).mean())
    ptr = Ztr @ v
    return {
        "v": torch.tensor(v, dtype=torch.float32),
        "p05": float(np.percentile(ptr, 5)),
        "p95": float(np.percentile(ptr, 95)),
        "val_select_acc": acc,
    }


# ---------------------------------------------------------------------- figures
def _save_grid(images, path, nrow):
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image((images.clamp(-1, 1) + 1) / 2, str(path), nrow=nrow, padding=2)


@torch.no_grad()
def _traversal_figures(model, editors, Zf, ds_sel, device, out_dir: Path, n_faces=4):
    D = Zf.shape[1]
    var = Zf.var(0)
    if D > MAX_TRAVERSAL_COORDS:
        coords = np.sort(np.argsort(-var)[:MAX_TRAVERSAL_COORDS])
        note = f"top {MAX_TRAVERSAL_COORDS} of {D} coordinates by variance of posterior means"
    else:
        coords = np.arange(D)
        note = f"all {D} coordinates"
    scale = np.sqrt(np.maximum(var, 1e-12))
    steps = np.linspace(-TRAVERSAL_RANGE, TRAVERSAL_RANGE, N_TRAVERSAL_VALUES)
    n = min(n_faces, len(ds_sel))
    x = torch.stack([ds_sel[i][0] for i in range(n)]).to(device)
    mu, _ = model.encode(x)
    files = []
    for f in range(n):
        rows = []
        for c in coords:
            z = mu[f : f + 1].repeat(N_TRAVERSAL_VALUES, 1)
            z[:, c] = torch.tensor(Zf[:, c].mean() + steps * scale[c], dtype=torch.float32, device=device)
            rows.append(model.decode(z))
        p = out_dir / f"traversal_face{f}.png"
        _save_grid(torch.cat(rows).cpu(), p, N_TRAVERSAL_VALUES)
        files.append(p.name)
    # five-level edit along each of the three directions, per face
    rows = []
    for f in range(n):
        for ed in editors.values():
            rows.append(ed.edit_levels(x[f : f + 1], 5)[0].cpu())
    p = out_dir / "direction_edits.png"
    _save_grid(torch.cat(rows), p, 5)
    files.append(p.name)
    return {
        "coordinates_shown": [int(c) for c in coords],
        "coordinates_note": note,
        "sweep": f"mean + a * std of posterior means, a in linspace(-{TRAVERSAL_RANGE}, {TRAVERSAL_RANGE}, {N_TRAVERSAL_VALUES})",
        "files": files,
        "direction_edit_row_order": [[f, name] for f in range(n) for name in editors],
    }, scale


@torch.no_grad()
def _coordinate_response(model, Zf, yf, ds_sel, scale, device, evaluator_path, n_images=64):
    D = Zf.shape[1]
    if evaluator_path is None:
        return "label_correlation", coordinate_correlations(Zf, yf).tolist()
    from src.evaluators import load_attribute_evaluator

    ev = load_attribute_evaluator(evaluator_path, device)
    ev.eval()
    k = ev.attr_names.index("Smiling")
    n = min(n_images, len(ds_sel))
    x = torch.stack([ds_sel[i][0] for i in range(n)]).to(device)
    mu, _ = model.encode(x)
    center = torch.tensor(Zf.mean(0), dtype=torch.float32, device=device)
    resp = []
    for c in range(D):
        probs = []
        for sgn in (-1.0, 1.0):
            z = mu.clone()
            z[:, c] = center[c] + sgn * TRAVERSAL_RANGE * float(scale[c])
            probs.append(torch.sigmoid(ev(model.decode(z))[:, k]))
        resp.append(float((probs[1] - probs[0]).mean()))
    return "evaluator_smile_probability_change", resp


# ---------------------------------------------------------------------- main API
def discover(run, data_root=None, evaluator_path=None, out_dir="results", max_train=20000) -> dict:
    device = get_device()
    model, cfg = load_model(run, device)
    if cfg["model"]["type"] != "vae":
        raise ValueError("Smile-direction discovery applies only to plain VAEs (model.type == 'vae'); "
                         f"this run is {cfg['model']['type']!r}.")
    model.eval()
    p = Path(run)
    run_dir = p if p.is_dir() else p.parent
    root = data_root or cfg["train"]["data_root"]
    tag = f"{cfg['name']}_seed{cfg['train']['seed']}"

    ds_fit = FaceDataset(root, split="val_fit")
    ds_sel = FaceDataset(root, split="val_select")
    ds_tr = FaceDataset(root, split="train", subset=max_train)
    Zf, yf = encode_split(model, ds_fit, device)
    Zs, ys = encode_split(model, ds_sel, device)
    # train labels are never read: only the latents are used for p05/p95
    Ztr, _ = encode_split(model, ds_tr, device)

    def make(Zf_, yf_):
        mean_diff = _entry(fit_mean_diff(Zf_, yf_), Ztr, Zf_, yf_, Zs, ys)
        probe = _entry(fit_probe(Zf_, yf_), Ztr, Zf_, yf_, Zs, ys)
        return mean_diff, probe

    mean_diff, probe = make(Zf, yf)
    v_c, idx = fit_coordinate(Zf, yf)
    coord = _entry(v_c, Ztr, Zf, yf, Zs, ys)
    coord["index"] = int(idx)

    label_eff = {}
    for size in LABEL_SIZES:
        n = len(yf) if size == "all" else size
        if size != "all" and n > len(yf):
            continue
        sel = balanced_first(yf, n) if size != "all" else np.arange(len(yf))
        if len(np.unique(yf[sel])) < 2:
            continue
        md, pr = make(Zf[sel], yf[sel])
        label_eff[str(size)] = {"mean_diff": md, "probe": pr}

    directions = {"mean_diff": mean_diff, "probe": probe, "coordinate": coord, "label_efficiency": label_eff}
    torch.save(directions, run_dir / "directions.pt")

    # figures
    out = Path(out_dir)
    trav_dir = out / "latent_traversals" / tag
    editors = {name: SmileEditor(model, "discovered", device, directions[name], name)
               for name in ("mean_diff", "probe", "coordinate")}
    trav_info, scale = _traversal_figures(model, editors, Zf, ds_sel, device, trav_dir)

    method, response = _coordinate_response(model, Zf, yf, ds_sel, scale, device, evaluator_path)
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run": str(run_dir),
        "latent_dim": int(Zf.shape[1]),
        "method": method,
        "response": response,
        "best_coordinate_by_label_correlation": int(idx),
        "traversal": trav_info,
        "val_select_acc": {k: directions[k]["val_select_acc"] for k in ("mean_diff", "probe", "coordinate")},
        "n_val_fit": int(len(yf)),
        "n_val_select": int(len(ys)),
        "n_train_calibration": int(len(Ztr)),
    }
    with open(metrics_dir / f"{tag}_coordinates.json", "w") as f:
        json.dump(report, f, indent=2)

    _print_summary(tag, directions)
    return directions


def _print_summary(tag, d):
    print(f"\nSmile directions for {tag}")
    print(f"{'direction':<22}{'val_select_acc':>15}{'p05':>10}{'p95':>10}")
    for k in ("mean_diff", "probe", "coordinate"):
        e = d[k]
        extra = f" (axis {e['index']})" if k == "coordinate" else ""
        print(f"{k + extra:<22}{e['val_select_acc']:>15.3f}{e['p05']:>10.2f}{e['p95']:>10.2f}")
    for size, ent in d["label_efficiency"].items():
        for k in ("mean_diff", "probe"):
            e = ent[k]
            print(f"{k + ' @' + size:<22}{e['val_select_acc']:>15.3f}{e['p05']:>10.2f}{e['p95']:>10.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--evaluator", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-train", type=int, default=20000)
    a = ap.parse_args(argv)
    try:
        discover(a.run, a.data_root, a.evaluator, a.out, a.max_train)
    except ValueError as e:
        if "applies only to plain VAEs" in str(e):
            sys.exit(f"error: {e}")
        raise


if __name__ == "__main__":
    main()
