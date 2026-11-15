"""Evaluate smile editors (test split only) and aggregate across runs/seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torchvision.utils import make_grid, save_image  # noqa: E402

from src import metrics as M  # noqa: E402
from src.data import FaceDataset, make_loader, to_unit  # noqa: E402
from src.evaluators import load_attribute_evaluator  # noqa: E402
from src.inference import SmileEditor  # noqa: E402
from src.utils import get_device  # noqa: E402

N_LEVELS = 5
SWEEP_T = [float(t) for t in np.linspace(-0.5, 1.5, 9)]


# --------------------------------------------------------------------------- helpers

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _mean_std(a, axis=0):
    a = np.asarray(a, dtype=np.float64)
    return a.mean(axis=axis), a.std(axis=axis)


def _run_dir(run) -> Path:
    run = Path(run)
    return run if run.is_dir() else run.parent


def _load_cfg(run) -> dict:
    d = _run_dir(run)
    p = d / "config.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text())
    ck = Path(run) if Path(run).is_file() else (d / "best.pt" if (d / "best.pt").exists() else d / "last.pt")
    return torch.load(ck, map_location="cpu", weights_only=False)["config"]


def _probs(evaluator, x):
    with torch.no_grad():
        return torch.sigmoid(evaluator(x)).cpu().numpy()


def _levels_probs(evaluator, levels):
    B, L = levels.shape[:2]
    return _probs(evaluator, levels.flatten(0, 1)).reshape(B, L, -1)


def _probe_accuracy(Xfit, yfit, Xtest, ytest) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    yfit, ytest = np.asarray(yfit).astype(int), np.asarray(ytest).astype(int)
    if len(np.unique(yfit)) < 2:
        return {"skipped": True, "reason": "val_fit has a single smile class"}
    lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)).fit(Xfit, yfit)
    mlp = make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=300, random_state=0)).fit(Xfit, yfit)
    maj = int(np.bincount(yfit).argmax())
    return {"linear_acc": float(lin.score(Xtest, ytest)), "mlp_acc": float(mlp.score(Xtest, ytest)),
            "majority_baseline_acc": float((ytest == maj).mean()), "n_fit": int(len(yfit)), "n_test": int(len(ytest))}


def _pair_similarity(emb, idx_pairs):
    if len(idx_pairs) == 0:
        return {"mean": None, "std": None, "n_pairs": 0}
    p = np.asarray(idx_pairs)
    s = (emb[p[:, 0]] * emb[p[:, 1]]).sum(1)
    return {"mean": float(s.mean()), "std": float(s.std()), "n_pairs": int(len(p))}


def _identity_reference(emb, ids, smiling, n_pairs=2000, seed=0) -> dict:
    """Real-image reference similarities from the dataset's identity labels."""
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for i, k in enumerate(ids):
        groups.setdefault(int(k), []).append(i)
    same = [(a, b) for g in groups.values() for a in g for b in g if smiling[a] == 1 and smiling[b] == 0]
    if len(same) > n_pairs:
        same = [same[j] for j in rng.choice(len(same), n_pairs, replace=False)]
    diff = []
    n = len(ids)
    for _ in range(n_pairs * 5):
        if len(diff) >= n_pairs or n < 2:
            break
        a, b = rng.integers(0, n, 2)
        if ids[a] != ids[b]:
            diff.append((int(a), int(b)))
    return {"same_identity_one_smiling_one_not": _pair_similarity(emb, same),
            "different_identity": _pair_similarity(emb, diff)}


@torch.no_grad()
def _grid(editor, ds, positions, path, device):
    """Rows: original, reconstruction, five levels. Columns: the given test images."""
    x = torch.stack([ds[int(i)][0] for i in positions]).to(device)
    rows = [x, editor.reconstruct(x)] + [editor.edit_levels(x, N_LEVELS)[:, k] for k in range(N_LEVELS)]
    g = make_grid(to_unit(torch.cat(rows).cpu()), nrow=len(positions), padding=2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(g, str(path))
    return str(path)


def _smile_control(editor, evaluator, ds, smile_idx, device, batch_size=64) -> dict:
    scores = []
    for x, _, _ in make_loader(ds, batch_size):
        x = x.to(device)
        scores.append(_levels_probs(evaluator, editor.edit_levels(x, N_LEVELS))[:, :, smile_idx])
    s = np.concatenate(scores)
    m, sd = _mean_std(s)
    return {"prob_mean": m, "prob_std": sd, "monotonicity": M.monotonicity_metrics(s)}


def _label_efficiency(run, direction, evaluator, ds, smile_idx, device) -> dict:
    dpath = _run_dir(run) / "directions.pt"
    if not dpath.exists():
        return {"available": False, "reason": "no directions.pt"}
    d = torch.load(dpath, map_location="cpu", weights_only=False)
    le = d.get("label_efficiency")
    if not le:
        return {"available": False, "reason": "directions.pt has no label_efficiency"}
    method = direction if direction in ("probe", "mean_diff") else "probe"
    out = {"available": True, "method": method, "budgets": {}}
    for budget, entry in le.items():
        if method not in entry:
            continue
        ed = SmileEditor.load(run, device=device, direction=method, direction_entry=entry[method])
        out["budgets"][str(budget)] = _smile_control(ed, evaluator, ds, smile_idx, device)
    return out


# --------------------------------------------------------------------------- per-run evaluation

def evaluate_run(run, evaluator, identity, data_root=None, out_dir="results", max_images=None,
                 direction="probe", batch_size=64, n_grid=8, device=None) -> dict:
    device = device or get_device()
    cfg = _load_cfg(run)
    name, seed = cfg["name"], int(cfg["train"]["seed"])
    tag = f"{name}_seed{seed}"
    data_root = data_root or cfg["train"]["data_root"]
    out_dir = Path(out_dir)
    if isinstance(evaluator, (str, Path)):
        evaluator = load_attribute_evaluator(evaluator, device)
    evaluator = evaluator.to(device).eval()
    editor = SmileEditor.load(run, device=device, direction=direction)

    ds = FaceDataset(data_root, "test", subset=max_images)
    names = ds.attr_names
    si = ds.smile_index
    groups = M.attribute_groups(names)
    do_id = identity is not None
    T = len(SWEEP_T)

    acc = {k: [] for k in ["mse", "mae", "ssim", "levels_p", "recon_p", "id_orig", "id_rec", "id_recon_orig",
                           "orig_emb", "sw_gain", "sw_id", "sw_leak", "codes_c", "codes_f", "labels", "rows"]}
    with torch.no_grad():
        for x, y, rows in make_loader(ds, batch_size):
            x = x.to(device)
            rec = editor.reconstruct(x)
            lev = editor.edit_levels(x, N_LEVELS)
            xu, ru = to_unit(x).cpu(), to_unit(rec).cpu()
            acc["mse"].append(M.recon_mse(xu, ru))
            acc["mae"].append(M.recon_mae(xu, ru))
            acc["ssim"].append(M.recon_ssim(xu, ru))
            lp = _levels_probs(evaluator, lev)
            rp = _probs(evaluator, rec)
            acc["levels_p"].append(lp)
            acc["recon_p"].append(rp)
            sw_p = np.stack([_probs(evaluator, editor.edit_at(x, t)) for t in SWEEP_T], 1)  # [B,T,40]
            acc["sw_gain"].append(sw_p[:, :, si] - rp[:, None, si])
            acc["sw_leak"].append(np.stack(
                [M.mean_abs_change(sw_p[:, k], rp, groups["unrelated"]) for k in range(T)], 1))
            if do_id:
                oe = identity.embed(x).cpu().numpy()
                re = identity.embed(rec).cpu().numpy()
                le = identity.embed(lev.flatten(0, 1)).cpu().numpy().reshape(x.shape[0], N_LEVELS, -1)
                acc["orig_emb"].append(oe)
                acc["id_recon_orig"].append(M.cosine_similarity(oe, re))
                acc["id_orig"].append(np.stack([M.cosine_similarity(oe, le[:, k]) for k in range(N_LEVELS)], 1))
                acc["id_rec"].append(np.stack([M.cosine_similarity(re, le[:, k]) for k in range(N_LEVELS)], 1))
                sw_id = []
                for t in SWEEP_T:
                    se = identity.embed(editor.edit_at(x, t)).cpu().numpy()
                    sw_id.append(M.cosine_similarity(re, se))
                acc["sw_id"].append(np.stack(sw_id, 1))
            acc["codes_c"].append(editor.content_code(x).cpu().numpy())
            acc["codes_f"].append(editor.encode(x).cpu().numpy())
            acc["labels"].append(y[:, si].numpy())
            acc["rows"].append(np.asarray(rows))
    cat = {k: (np.concatenate(v) if v else None) for k, v in acc.items()}
    n = len(cat["mse"])
    lp, rp = cat["levels_p"], cat["recon_p"]
    smile = lp[:, :, si]

    res = {
        "name": name, "seed": seed, "run": str(run), "kind": editor.kind, "direction": direction,
        "image_size": ds.image_size, "n_images": n, "split": "test",
        "evaluator": dict(getattr(evaluator, "info", {})),
        "reconstruction": {"mse": cat["mse"].mean(), "mae": cat["mae"].mean(), "ssim": cat["ssim"].mean()},
    }
    pm, ps = _mean_std(smile)
    res["smile_control"] = {"levels": list(range(1, N_LEVELS + 1)), "prob_mean": pm, "prob_std": ps,
                            "monotonicity": M.monotonicity_metrics(smile)}

    # identity
    if do_id:
        io, ir = cat["id_orig"], cat["id_rec"]
        res["identity"] = {
            "skipped": False,
            "embedder_pretrained": bool(getattr(identity, "pretrained", True)),
            "to_original": {"mean": io.mean(0), "std": io.std(0)},
            "to_reconstruction": {"mean": ir.mean(0), "std": ir.std(0)},
            "reconstruction_to_original": {"mean": cat["id_recon_orig"].mean(), "std": cat["id_recon_orig"].std()},
            "reference_real_images": _identity_reference(cat["orig_emb"], ds.identity, ds.attrs[:, si]),
        }
    else:
        res["identity"] = {"skipped": True, "reason": "--no-identity"}

    # leakage
    leak = {}
    for g in ("unrelated", "smile_entailed"):
        idx = groups[g]
        lv = M.mean_abs_change(lp[:, -1], lp[:, 0], idx)
        vs = np.stack([M.mean_abs_change(lp[:, k], rp, idx) for k in range(N_LEVELS)], 1)
        leak[g] = {"attributes": [names[j] for j in idx], "level1_to_level5": lv.mean(),
                   "level_vs_reconstruction": vs.mean(0)}
    leak["per_attribute_level1_to_level5"] = {
        names[j]: float(np.abs(lp[:, -1, j] - lp[:, 0, j]).mean()) for j in range(len(names))}
    res["leakage"] = leak

    # trade-off sweep
    sg_m, sg_s = _mean_std(cat["sw_gain"])
    lk_m, lk_s = _mean_std(cat["sw_leak"])
    tr = {"t": SWEEP_T, "smile_gain": sg_m, "smile_gain_std": sg_s,
          "leakage_unrelated": lk_m, "leakage_unrelated_std": lk_s}
    if do_id:
        si_m, si_s = _mean_std(cat["sw_id"])
        tr.update(identity_sim_to_reconstruction=si_m, identity_sim_to_reconstruction_std=si_s)
    else:
        tr["identity_sim_to_reconstruction"] = {"skipped": True}
    res["tradeoff"] = tr

    # disentanglement: probes fit on val_fit, tested on test
    vf = FaceDataset(data_root, "val_fit", subset=5000)
    fc, ff, fl = [], [], []
    with torch.no_grad():
        for x, y, _ in make_loader(vf, batch_size):
            x = x.to(device)
            fc.append(editor.content_code(x).cpu().numpy())
            ff.append(editor.encode(x).cpu().numpy())
            fl.append(y[:, si].numpy())
    fc, ff, fl = np.concatenate(fc), np.concatenate(ff), np.concatenate(fl)
    res["disentanglement"] = {
        "content_code": _probe_accuracy(fc, fl, cat["codes_c"], cat["labels"]),
        "full_latent": _probe_accuracy(ff, fl, cat["codes_f"], cat["labels"]),
    }

    # label efficiency (discovered directions only)
    if editor.kind == "discovered":
        res["label_efficiency"] = _label_efficiency(run, direction, evaluator, ds, si, device)
    else:
        res["label_efficiency"] = {"available": False, "reason": "supervised run"}

    # grids and failure cases
    k = min(n_grid, n)
    gain = smile[:, -1] - smile[:, 0]
    fail = {}
    worst = np.argsort(gain)[:k]
    fail["lowest_smile_gain"] = {
        "test_rows": cat["rows"][worst], "values": gain[worst],
        "grid": _grid(editor, ds, worst, out_dir / "smile_edits" / f"{tag}_worst_smile_gain.png", device)}
    if do_id:
        idm = cat["id_orig"].mean(1)
        worst_id = np.argsort(idm)[:k]
        fail["lowest_identity_similarity"] = {
            "criterion": "mean cosine similarity to the original across the five levels",
            "test_rows": cat["rows"][worst_id], "values": idm[worst_id],
            "grid": _grid(editor, ds, worst_id, out_dir / "smile_edits" / f"{tag}_worst_identity.png", device)}
    else:
        fail["lowest_identity_similarity"] = {"skipped": True}
    fail["first_images"] = {"grid": _grid(editor, ds, np.arange(k), out_dir / "smile_edits" / f"{tag}_first.png",
                                          device)}
    res["failure_cases"] = fail
    rec_path = out_dir / "reconstructions" / f"{tag}.png"
    with torch.no_grad():
        xs = torch.stack([ds[i][0] for i in range(k)]).to(device)
        rg = make_grid(to_unit(torch.cat([xs, editor.reconstruct(xs)]).cpu()), nrow=k, padding=2)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(rg, str(rec_path))
    res["files"] = {"reconstructions": str(rec_path)}

    fig_path = out_dir / "figures" / f"{tag}_tradeoff.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_tradeoff({name: [res]}, fig_path)
    res["files"]["tradeoff_figure"] = str(fig_path)

    mdir = out_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    res = _jsonable(res)
    (mdir / f"{tag}.json").write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------------- aggregation

def _get(d, path):
    for p in path:
        try:
            d = d[p]
        except (KeyError, IndexError, TypeError):
            return None
    return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None


# (column, path, reliability family)
FIELDS = [
    ("recon_mse", ("reconstruction", "mse"), None),
    ("recon_mae", ("reconstruction", "mae"), None),
    ("recon_ssim", ("reconstruction", "ssim"), None),
    *[(f"smile_prob_level{k + 1}", ("smile_control", "prob_mean", k), "smile") for k in range(N_LEVELS)],
    ("monotonic_fraction", ("smile_control", "monotonicity", "monotonic_fraction"), "smile"),
    ("spearman_mean", ("smile_control", "monotonicity", "spearman_mean"), "smile"),
    ("score_range_1to5", ("smile_control", "monotonicity", "score_range_mean"), "smile"),
    ("identity_to_original_level1", ("identity", "to_original", "mean", 0), "identity"),
    ("identity_to_original_level5", ("identity", "to_original", "mean", 4), "identity"),
    ("identity_to_recon_level1", ("identity", "to_reconstruction", "mean", 0), "identity"),
    ("identity_to_recon_level5", ("identity", "to_reconstruction", "mean", 4), "identity"),
    ("identity_ref_same_id_diff_smile",
     ("identity", "reference_real_images", "same_identity_one_smiling_one_not", "mean"), "identity"),
    ("identity_ref_different_id", ("identity", "reference_real_images", "different_identity", "mean"), "identity"),
    ("leakage_unrelated_1to5", ("leakage", "unrelated", "level1_to_level5"), "unrelated_attributes"),
    ("leakage_unrelated_level5_vs_recon", ("leakage", "unrelated", "level_vs_reconstruction", 4),
     "unrelated_attributes"),
    ("change_smile_entailed_1to5", ("leakage", "smile_entailed", "level1_to_level5"), "smile_entailed_attributes"),
    ("change_smile_entailed_level5_vs_recon", ("leakage", "smile_entailed", "level_vs_reconstruction", 4),
     "smile_entailed_attributes"),
    ("probe_content_linear_acc", ("disentanglement", "content_code", "linear_acc"), None),
    ("probe_content_mlp_acc", ("disentanglement", "content_code", "mlp_acc"), None),
    ("probe_full_linear_acc", ("disentanglement", "full_latent", "linear_acc"), None),
    ("probe_full_mlp_acc", ("disentanglement", "full_latent", "mlp_acc"), None),
]
TABLES = [
    ("Reconstruction", ["recon_mse", "recon_mae", "recon_ssim"]),
    ("Smile control (evaluator smile probability per level)",
     [f"smile_prob_level{k}" for k in range(1, 6)] + ["monotonic_fraction", "spearman_mean", "score_range_1to5"]),
    ("Identity (cosine similarity)", ["identity_to_original_level1", "identity_to_original_level5",
                                      "identity_to_recon_level1", "identity_to_recon_level5",
                                      "identity_ref_same_id_diff_smile", "identity_ref_different_id"]),
    ("Attribute change (mean absolute change in evaluator probability; smile-entailed reported separately)",
     ["leakage_unrelated_1to5", "leakage_unrelated_level5_vs_recon", "change_smile_entailed_1to5",
      "change_smile_entailed_level5_vs_recon"]),
    ("Smile information in the content code (probe accuracy on test; lower is better disentanglement)",
     ["probe_content_linear_acc", "probe_content_mlp_acc", "probe_full_linear_acc", "probe_full_mlp_acc"]),
]


def _fmt(m, s, n):
    if m is None:
        return "n/a"
    return f"{m:.4f} ± {s:.4f}" if s is not None else f"{m:.4f} (n={n})"


def _plot_tradeoff(by_model: dict, path):
    """by_model: name -> list of per-run result dicts. One curve per model, seed-averaged with a band."""
    def stack(runs, key):
        vals = [r["tradeoff"].get(key) for r in runs]
        if any(v is None or isinstance(v, dict) for v in vals):
            return None
        return np.array(vals, dtype=float)

    has_id = any(stack(r, "identity_sim_to_reconstruction") is not None for r in by_model.values())
    fig, axes = plt.subplots(1, 2 if has_id else 1, figsize=(11 if has_id else 5.5, 4.4), squeeze=False)
    axes = axes[0]
    panels = ([("identity_sim_to_reconstruction", "identity similarity to reconstruction")] if has_id else []) + \
             [("leakage_unrelated", "unrelated-attribute leakage")]
    for ax, (key, label) in zip(axes, panels):
        for ci, (name, runs) in enumerate(by_model.items()):
            x, y = stack(runs, "smile_gain"), stack(runs, key)
            if x is None or y is None:
                continue
            c = f"C{ci}"
            xm, ym = x.mean(0), y.mean(0)
            ax.plot(xm, ym, "-o", ms=3, color=c, label=f"{name} (n={len(runs)})")
            if len(runs) > 1:
                ax.fill_between(xm, ym - y.std(0), ym + y.std(0), color=c, alpha=0.2)
        ax.set_xlabel("smile probability gain over reconstruction")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Matched-strength trade-off sweep (t from -0.5 to 1.5)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_levels(by_model: dict, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for ci, (name, runs) in enumerate(by_model.items()):
        y = np.array([r["smile_control"]["prob_mean"] for r in runs], dtype=float)
        x = np.arange(1, y.shape[1] + 1)
        ax.plot(x, y.mean(0), "-o", color=f"C{ci}", label=f"{name} (n={len(runs)})")
        if len(runs) > 1:
            ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=f"C{ci}", alpha=0.2)
    ax.set_xlabel("smile level")
    ax.set_ylabel("evaluator smile probability")
    ax.set_xticks(range(1, N_LEVELS + 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def aggregate(results_dir="results", run_jsons=None) -> dict:
    """Mean ± std over seeds per model name. Writes summary.csv, summary.md and the two figures."""
    results_dir = Path(results_dir)
    mdir = results_dir / "metrics"
    if run_jsons is None:
        run_jsons = [p for p in sorted(mdir.glob("*_seed*.json")) if "evaluator_validation" not in p.name]
    runs = [json.loads(Path(p).read_text()) for p in run_jsons]
    runs = [r for r in runs if "name" in r and "tradeoff" in r]
    by_model: dict = {}
    for r in runs:
        by_model.setdefault(r["name"], []).append(r)

    val_path = mdir / "evaluator_validation.json"
    rel = json.loads(val_path.read_text()).get("reliability") if val_path.exists() else None

    def reliable(family):
        if family is None:
            return None
        return None if rel is None else rel.get(family)

    rows = []
    for name, rs in by_model.items():
        row = {"model": name, "n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs)}
        for col, path, fam in FIELDS:
            vals = [v for v in (_get(r, path) for r in rs) if v is not None]
            row[col] = (float(np.mean(vals)) if vals else None,
                        float(np.std(vals, ddof=1)) if len(vals) > 1 else None, len(vals))
        rows.append(row)

    mdir.mkdir(parents=True, exist_ok=True)
    fam_of = {c: f for c, _, f in FIELDS}
    with open(results_dir / "metrics" / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "n_seeds"]
        for col, _, _ in FIELDS:
            header += [f"{col}_mean", f"{col}_std", f"{col}_n"]
        header += [f"evaluator_reliable_{k}" for k in ("smile", "unrelated_attributes", "smile_entailed_attributes",
                                                       "identity")]
        w.writerow(header)
        for row in rows:
            line = [row["model"], row["n_seeds"]]
            for col, _, _ in FIELDS:
                m, s, n = row[col]
                line += ["" if m is None else m, "" if s is None else s, n]
            line += ["" if rel is None or rel.get(k) is None else rel.get(k)
                     for k in ("smile", "unrelated_attributes", "smile_entailed_attributes", "identity")]
            w.writerow(line)

    md = ["# Summary", "", "Mean ± std across seeds per model (sample std; n=1 shows no std). `n/a` = not computed "
          "(for example identity skipped).", ""]
    if rel is None:
        md += ["Evaluator validation file not found: reliability of evaluator-based metrics is unknown.", ""]
    else:
        th = rel.get("thresholds", {})
        flags = {k: rel.get(k) for k in ("smile", "unrelated_attributes", "smile_entailed_attributes", "identity")}
        md += ["Evaluator reliability (from evaluator_validation.json; thresholds " + json.dumps(th) + "): " +
               ", ".join(f"{k}={v}" for k, v in flags.items()), ""]
        bad = [k for k, v in flags.items() if v is False]
        if bad:
            md += [f"WARNING: unreliable metric families, marked with † below: {', '.join(bad)}.", ""]
    for title, cols in TABLES:
        md += [f"## {title}", ""]
        heads = []
        for c in cols:
            heads.append(c + (" †" if reliable(fam_of[c]) is False else ""))
        md += ["| model | seeds | " + " | ".join(heads) + " |", "|" + "---|" * (len(cols) + 2)]
        for row in rows:
            md.append(f"| {row['model']} | {row['n_seeds']} | " +
                      " | ".join(_fmt(*row[c]) for c in cols) + " |")
        md.append("")
    (results_dir / "metrics" / "summary.md").write_text("\n".join(md))

    fdir = results_dir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    if by_model:
        _plot_tradeoff(by_model, fdir / "tradeoff.png")
        _plot_levels(by_model, fdir / "smile_by_level.png")
    return {"rows": rows, "reliability": rel}


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate smile editing runs on the test split.")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--evaluator", default="runs/evaluator/attr.pt")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--direction", default="probe")
    ap.add_argument("--no-identity", action="store_true")
    a = ap.parse_args(argv)
    device = get_device()
    evaluator = load_attribute_evaluator(a.evaluator, device)
    identity = None
    if not a.no_identity:
        from src.evaluators import IdentityEmbedder
        identity = IdentityEmbedder(device, pretrained=True)
    for run in a.runs:
        r = evaluate_run(run, evaluator, identity, a.data_root, a.out, a.max_images, a.direction, device=device)
        print(f"evaluated {r['name']} seed{r['seed']}: smile levels {np.round(r['smile_control']['prob_mean'], 3)}")
    aggregate(a.out)


if __name__ == "__main__":
    main()
