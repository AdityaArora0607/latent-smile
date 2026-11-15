"""Training loop for both approaches.

    python -m src.train --config configs/baseline.yaml [--seed K] [--subset N] [--epochs E]
                        [--batch-size B] [--data-root P] [--out-dir P] [--resume]
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402
from tqdm import tqdm  # noqa: E402

from .checkpoint import save_checkpoint  # noqa: E402
from .config import apply_overrides, fill_defaults, load_config  # noqa: E402
from .data import FaceDataset, make_loader  # noqa: E402
from .losses import supervised_loss, vae_loss  # noqa: E402
from .models import build_model  # noqa: E402
from .utils import get_device, seed_everything  # noqa: E402

ACTIVE_KL_THRESHOLD = 0.01
SENSITIVITY_BATCH = 64
RECON_GRID = 8


def run_dir_for(cfg: dict) -> Path:
    t = cfg["train"]
    return Path(t["out_dir"]) / cfg["name"] / f"seed{t['seed']}"


def ramp(epoch_float: float, warmup_epochs: float) -> float:
    """Linear 0 -> 1 over ``warmup_epochs`` epochs (1 if warmup is 0)."""
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, epoch_float / warmup_epochs)))


@torch.no_grad()
def smile_sensitivity(model, x) -> float:
    """Mean |pixel change| when z_smile goes from -m to +m, content at the posterior mean."""
    was_training = model.training
    model.eval()
    mu, _ = model.encode(x)
    m = model.smile_prior_mean
    z_pos, z_neg = mu.clone(), mu.clone()
    z_pos[:, -1], z_neg[:, -1] = m, -m
    diff = (model.decode(z_pos) - model.decode(z_neg)).abs().mean().item()
    model.train(was_training)
    return diff


def _is_supervised(cfg) -> bool:
    return cfg["model"]["type"] == "supervised"


def compute_loss(model, batch, cfg, beta, lambd, smile_index, device):
    """One forward pass + loss on a batch."""
    lc = cfg["loss"]
    if _is_supervised(cfg):
        x, attrs, _ = batch
        smiling = attrs[:, smile_index].to(device, dtype=torch.float32)
        x = x.to(device, dtype=torch.float32)
        out = model(x)
        return supervised_loss(model, out, x, smiling, lc["sigma"], beta, lc["adv_weight"], lambd)
    # type: vae -- LABELS ARE NEVER READ. Only the image (batch[0]) is touched; the attrs
    # tensor (batch[1]) and index (batch[2]) are ignored on purpose.
    x = batch[0].to(device, dtype=torch.float32)
    out = model(x)
    return vae_loss(out, x, lc["sigma"], beta)


def run_epoch(model, loader, cfg, device, smile_index, epoch_idx, optimizer=None, desc=""):
    """One pass over ``loader``. Training when ``optimizer`` is given, else evaluation at full beta."""
    lc = cfg["loss"]
    training = optimizer is not None
    supervised = _is_supervised(cfg)
    model.train(training)
    keys = ["loss", "recon", "kl"] + (["adv", "adv_acc", "smile_acc"] if supervised else [])
    sums = {k: 0.0 for k in keys}
    kl_dim_sum, n_seen = None, 0
    n_steps = len(loader)
    bar = tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty())
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(bar):
            if training:
                pos = epoch_idx + step / max(n_steps, 1)
                beta = lc["beta"] * ramp(pos, lc["kl_warmup_epochs"])
                lambd = ramp(pos, lc["adv_warmup_epochs"])
            else:
                beta, lambd = lc["beta"], 1.0
            res = compute_loss(model, batch, cfg, beta, lambd, smile_index, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                res["loss"].backward()
                optimizer.step()
            bs = batch[0].shape[0]
            n_seen += bs
            for k in keys:
                sums[k] += res[k].detach().item() * bs
            kd = res["kl_per_dim"].detach().cpu() * bs  # CPU before logging (MPS care)
            kl_dim_sum = kd if kl_dim_sum is None else kl_dim_sum + kd
    stats = {k: v / n_seen for k, v in sums.items()}
    kl_per_dim = kl_dim_sum / n_seen
    stats["active_dims"] = int((kl_per_dim > ACTIVE_KL_THRESHOLD).sum().item())
    if not training:
        # Selection metric: ELBO terms at full beta (the adversary term is excluded).
        stats["select_loss"] = stats["recon"] + lc["beta"] * stats["kl"]
    return stats


# ---------------------------------------------------------------- logging / plots

def _log_columns(supervised: bool) -> list[str]:
    cols = ["epoch", "train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl",
            "active_dims", "beta"]
    if supervised:
        cols += ["lambd", "train_adv", "train_adv_acc", "val_adv", "val_adv_acc", "val_smile_acc", "smile_sensitivity"]
    return cols + ["seconds"]


def _read_log(path: Path, max_epoch: int) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["epoch"] = int(r["epoch"])
    return [r for r in rows if r["epoch"] <= max_epoch]


def _write_log(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _append_log(path: Path, cols: list[str], row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writerow(row)


def plot_curves(rows: list[dict], supervised: bool, path: Path) -> None:
    ep = [r["epoch"] for r in rows]
    panels = [("Loss", "loss"), ("Reconstruction", "recon"), ("KL", "kl")]
    n = len(panels) + (1 if supervised else 0)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4))
    for ax, (title, key) in zip(axes, panels):
        ax.plot(ep, [r[f"train_{key}"] for r in rows], label="train")
        ax.plot(ep, [r[f"val_{key}"] for r in rows], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    if supervised:
        ax = axes[-1]
        ax.plot(ep, [r["val_adv_acc"] for r in rows], label="adversary acc (val)")
        ax.plot(ep, [r["smile_sensitivity"] for r in rows], label="smile sensitivity")
        ax.set_title("Supervised diagnostics")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_recon(model, x, path: Path, device) -> None:
    model.eval()
    x = x.to(device, dtype=torch.float32)
    mu, _ = model.encode(x)
    recon = model.decode(mu)
    grid = make_grid(torch.cat([x, recon]).cpu().clamp(-1, 1) * 0.5 + 0.5, nrow=x.shape[0], padding=2)
    fig, ax = plt.subplots(figsize=(1.4 * x.shape[0], 3.2))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_title("validation: originals (top) / reconstructions (bottom)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main entry

def train(cfg: dict, resume: bool = False) -> Path:
    cfg = fill_defaults(copy.deepcopy(cfg))
    t = cfg["train"]
    supervised = _is_supervised(cfg)
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed_everything(t["seed"])
    device = get_device()

    train_ds = FaceDataset(t["data_root"], "train", subset=t["subset"], flip=t["flip"], seed=t["seed"])
    val_ds = FaceDataset(t["data_root"], "val_select", seed=t["seed"])
    if train_ds.image_size != cfg["model"]["image_size"]:
        raise ValueError(f"dataset image size {train_ds.image_size} != model.image_size "
                         f"{cfg['model']['image_size']}")
    smile_index = train_ds.smile_index
    train_loader = make_loader(train_ds, t["batch_size"], shuffle=True, num_workers=t["num_workers"])
    val_loader = make_loader(val_ds, t["batch_size"], shuffle=False, num_workers=t["num_workers"])

    # Fixed validation images for the diagnostics and the reconstruction grid.
    fixed_x = torch.stack([val_ds[i][0] for i in range(min(SENSITIVITY_BATCH, len(val_ds)))]).to(device)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
    cols = _log_columns(supervised)
    log_path = run_dir / "log.csv"
    start_epoch, best_val, rows = 0, float("inf"), []

    last = run_dir / "last.pt"
    if resume and last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"])
        if ckpt.get("best_val") is not None:
            best_val = float(ckpt["best_val"])
        rows = _read_log(log_path, start_epoch)
        _write_log(log_path, cols, rows)
        seed_everything(t["seed"] + start_epoch)  # fresh data order after resuming
        print(f"resumed {run_dir} from epoch {start_epoch}", flush=True)
    else:
        if resume:
            print(f"--resume given but {last} not found; starting fresh", flush=True)
        _write_log(log_path, cols, [])

    for epoch_idx in range(start_epoch, t["epochs"]):
        t0 = time.time()
        tr = run_epoch(model, train_loader, cfg, device, smile_index, epoch_idx, optimizer,
                       desc=f"epoch {epoch_idx + 1}/{t['epochs']}")
        va = run_epoch(model, val_loader, cfg, device, smile_index, epoch_idx, None, desc="val")
        pos_end = float(epoch_idx + 1)
        row = {
            "epoch": epoch_idx + 1,
            "train_loss": tr["loss"], "train_recon": tr["recon"], "train_kl": tr["kl"],
            "val_loss": va["select_loss"], "val_recon": va["recon"], "val_kl": va["kl"],
            "active_dims": va["active_dims"],
            "beta": cfg["loss"]["beta"] * ramp(pos_end, cfg["loss"]["kl_warmup_epochs"]),
        }
        if supervised:
            row.update({
                "lambd": ramp(pos_end, cfg["loss"]["adv_warmup_epochs"]),
                "train_adv": tr["adv"], "train_adv_acc": tr["adv_acc"],
                "val_adv": va["adv"], "val_adv_acc": va["adv_acc"],
                "val_smile_acc": va["smile_acc"],
                "smile_sensitivity": smile_sensitivity(model, fixed_x),
            })
        row["seconds"] = time.time() - t0
        rows.append(row)
        _append_log(log_path, cols, row)

        if va["select_loss"] < best_val:
            best_val = va["select_loss"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch_idx + 1, cfg, best_val)
        save_checkpoint(last, model, optimizer, epoch_idx + 1, cfg, best_val)
        msg = (f"epoch {epoch_idx + 1}/{t['epochs']} train {row['train_loss']:.2f} "
               f"val {row['val_loss']:.2f} recon {row['val_recon']:.2f} kl {row['val_kl']:.2f} "
               f"active {row['active_dims']}")
        if supervised:
            msg += f" adv_acc {row['val_adv_acc']:.3f} smile_acc {row['val_smile_acc']:.3f} smile_sens {row['smile_sensitivity']:.4f}"
        print(msg, flush=True)

    if rows:
        plot_curves(rows, supervised, run_dir / "curves.png")
    plot_recon(model, fixed_x[:RECON_GRID], run_dir / "recon.png", device)
    return run_dir


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--subset", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--data-root")
    p.add_argument("--out-dir")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    overrides = {k: v for k, v in {
        "train.seed": a.seed, "train.subset": a.subset, "train.epochs": a.epochs,
        "train.batch_size": a.batch_size, "train.data_root": a.data_root,
        "train.out_dir": a.out_dir}.items() if v is not None}
    cfg = load_config(a.config, overrides)
    return train(cfg, resume=a.resume)


if __name__ == "__main__":
    main()
