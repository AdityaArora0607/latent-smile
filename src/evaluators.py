"""Attribute evaluator (small CNN), identity embedder, and the evaluator training / validation pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.checkpoint import load_model
from src.data import FaceDataset, make_loader
from src.metrics import attribute_groups
from src.utils import get_device, seed_everything

SMILE_AUC_MIN = 0.90
ATTRIBUTE_GROUP_AUC_MIN = 0.80
IDENTITY_AUC_MIN = 0.90


class AttributeEvaluator(nn.Module):
    """5 stride-2 conv blocks, adaptive average pooling, linear head with one logit per attribute."""

    def __init__(self, attr_names, image_size: int = 64, width: int = 32):
        super().__init__()
        self.attr_names = list(attr_names)
        self.image_size = int(image_size)
        self.width = int(width)
        chans = [3, width, width * 2, width * 4, width * 4, width * 8]
        blocks = []
        for i in range(5):
            blocks += [
                nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(chans[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(chans[-1], len(self.attr_names)))
        self.info: dict = {}

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


def load_attribute_evaluator(path, device=None) -> AttributeEvaluator:
    device = device or get_device()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = AttributeEvaluator(ck["attr_names"], ck["image_size"], ck.get("width", 32))
    model.load_state_dict(ck["state"])
    model.info = dict(ck.get("info", {}))
    return model.to(device).eval()


class IdentityEmbedder:
    """facenet-pytorch InceptionResnetV1 (VGGFace2). pretrained=False uses random weights and never downloads."""

    def __init__(self, device=None, pretrained: bool = True):
        from facenet_pytorch import InceptionResnetV1

        self.device = device or get_device()
        self.pretrained = pretrained
        self.model = InceptionResnetV1(pretrained="vggface2" if pretrained else None).eval().to(self.device)

    @torch.no_grad()
    def embed(self, x, batch_size: int = 128) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            b = x[i:i + batch_size].to(self.device)
            b = F.interpolate(b, size=(160, 160), mode="bilinear", align_corners=False)
            outs.append(F.normalize(self.model(b), dim=1))
        return torch.cat(outs) if outs else torch.zeros(0, 512, device=self.device)


# --------------------------------------------------------------------------- helpers

def _reconstruct(model, x):
    mu, _ = model.encode(x)
    return model.decode(mu)


def _safe_auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def identity_verification_auc(emb: np.ndarray, ids: np.ndarray, n_pairs: int = 1000, seed: int = 0) -> dict:
    """AUC of cosine similarity for separating same-identity from different-identity pairs."""
    rng = np.random.default_rng(seed)
    n = len(ids)
    groups: dict = {}
    for i, k in enumerate(ids):
        groups.setdefault(int(k), []).append(i)
    same = [(a, b) for g in groups.values() for ai, a in enumerate(g) for b in g[ai + 1:]]
    if len(same) > n_pairs:
        same = [same[j] for j in rng.choice(len(same), n_pairs, replace=False)]
    diff = []
    for _ in range(n_pairs * 5):
        if len(diff) >= n_pairs:
            break
        a, b = rng.integers(0, n, 2)
        if ids[a] != ids[b]:
            diff.append((int(a), int(b)))
    out = {"n_same": len(same), "n_diff": len(diff), "auc": None}
    if same and diff:
        pairs = np.array(same + diff)
        sims = (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(1)
        labels = np.r_[np.ones(len(same)), np.zeros(len(diff))]
        out["auc"] = _safe_auc(labels, sims)
    return out


# --------------------------------------------------------------------------- training

def train_evaluator(data_root, out, recon_run=None, epochs: int = 10, subset=None, batch_size: int = 128,
                    lr: float = 1e-3, seed: int = 0, device=None, width: int = 32) -> Path:
    """Train on the train split: half of each batch real, half reconstructed by the evaluator-only VAE
    (or Gaussian-blurred when no recon run is given). Returns the path of attr.pt."""
    import torchvision.transforms.functional as TF

    device = device or get_device()
    seed_everything(seed)
    ds = FaceDataset(data_root, "train", subset=subset, flip=True, seed=seed)
    loader = make_loader(ds, batch_size, shuffle=True, drop_last=len(ds) > batch_size)
    recon_model = None
    if recon_run is not None:
        recon_model, _ = load_model(recon_run, device)
        recon_model.eval()
        augmentation = "half real, half reconstructions from evaluator-only VAE (computed on the fly)"
    else:
        augmentation = "real images plus Gaussian-blur augmentation on half of each batch (no recon run given)"
    model = AttributeEvaluator(ds.attr_names, ds.image_size, width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    last_loss = None
    for ep in range(epochs):
        model.train()
        tot, cnt = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            h = x.shape[0] // 2
            with torch.no_grad():
                if recon_model is not None:
                    second = _reconstruct(recon_model, x[h:])
                else:
                    second = TF.gaussian_blur(x[h:], kernel_size=5, sigma=float(rng.uniform(0.5, 2.0)))
                inp = torch.cat([x[:h], second])
            loss = F.binary_cross_entropy_with_logits(model(inp), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]
            cnt += x.shape[0]
        last_loss = tot / max(cnt, 1)
        print(f"[evaluator] epoch {ep + 1}/{epochs} loss {last_loss:.4f}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "attr.pt"
    torch.save({
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "attr_names": ds.attr_names,
        "image_size": ds.image_size,
        "width": width,
        "info": {"recon_run": str(recon_run) if recon_run is not None else None, "augmentation": augmentation,
                 "epochs": epochs, "train_subset": subset, "final_train_loss": last_loss, "seed": seed},
    }, path)
    return path


# --------------------------------------------------------------------------- validation

@torch.no_grad()
def validate_evaluator(evaluator_path, data_root, recon_run=None, results="results", identity=None,
                       subset=None, batch_size: int = 128, n_pairs: int = 1000, device=None) -> dict:
    """Validate on val_select (real images and reconstructions) and write results/metrics/evaluator_validation.json.
    identity: an IdentityEmbedder, or None to skip identity verification."""
    device = device or get_device()
    ev = load_attribute_evaluator(evaluator_path, device)
    ds = FaceDataset(data_root, "val_select", subset=subset)
    recon_model = load_model(recon_run, device)[0].eval() if recon_run is not None else None
    names = ev.attr_names
    smile = ds.smile_index
    real_p, rec_p, real_x, rec_x = [], [], [], []
    keep_imgs = identity is not None
    for x, _, _ in make_loader(ds, batch_size):
        x = x.to(device)
        real_p.append(torch.sigmoid(ev(x)).cpu())
        if keep_imgs:
            real_x.append(x.cpu())
        if recon_model is not None:
            r = _reconstruct(recon_model, x)
            rec_p.append(torch.sigmoid(ev(r)).cpu())
            if keep_imgs:
                rec_x.append(r.cpu())
    labels = ds.attrs.astype(np.int64)

    def per_attr(P):
        P = torch.cat(P).numpy()
        res = {}
        for j, n in enumerate(names):
            res[n] = {"accuracy": float(((P[:, j] > 0.5) == (labels[:, j] == 1)).mean()),
                      "auc": _safe_auc(labels[:, j], P[:, j])}
        return res

    attrs = {"real": per_attr(real_p)}
    if rec_p:
        attrs["reconstruction"] = per_attr(rec_p)
    sm = {k: {"accuracy": v[names[smile]]["accuracy"], "auc": v[names[smile]]["auc"]} for k, v in attrs.items()}
    n_img = len(ds)

    ident = {"skipped": identity is None}
    if identity is None:
        ident["reason"] = "--no-identity"
    else:
        ids = ds.identity
        ident["real"] = identity_verification_auc(identity.embed(torch.cat(real_x)).cpu().numpy(), ids, n_pairs)
        if rec_x:
            ident["reconstruction"] = identity_verification_auc(identity.embed(torch.cat(rec_x)).cpu().numpy(), ids, n_pairs)
        ident["pretrained"] = bool(getattr(identity, "pretrained", True))

    def worst_auc(getter):
        vals = [getter(k) for k in attrs]
        return None if any(v is None for v in vals) else min(vals)

    groups = attribute_groups(names)

    def group_auc(idx):
        per_set = []
        for k in attrs:
            a = [attrs[k][names[j]]["auc"] for j in idx]
            a = [v for v in a if v is not None]
            per_set.append(float(np.mean(a)) if a else None)
        return None if (not per_set or any(v is None for v in per_set)) else min(per_set)

    smile_auc = worst_auc(lambda k: sm[k]["auc"])
    unrel_auc, ent_auc = group_auc(groups["unrelated"]), group_auc(groups["smile_entailed"])
    id_aucs = [ident[k]["auc"] for k in ("real", "reconstruction") if k in ident]
    id_auc = None if (ident["skipped"] or not id_aucs or any(a is None for a in id_aucs)) else min(id_aucs)
    per_attr_rel = {}
    for j, n in enumerate(names):
        a = [attrs[k][n]["auc"] for k in attrs]
        per_attr_rel[n] = bool(all(v is not None and v >= ATTRIBUTE_GROUP_AUC_MIN for v in a))
    report = {
        "evaluator": {"path": str(evaluator_path), "image_size": ev.image_size, **ev.info},
        "split": "val_select",
        "n_images": n_img,
        "smile": sm,
        "attributes": attrs,
        "identity": ident,
        "reliability": {
            "thresholds": {"smile_auc_min": SMILE_AUC_MIN, "attribute_group_mean_auc_min": ATTRIBUTE_GROUP_AUC_MIN,
                           "per_attribute_auc_min": ATTRIBUTE_GROUP_AUC_MIN, "identity_auc_min": IDENTITY_AUC_MIN},
            "basis": "minimum over real images and reconstructions (real only if no recon run)",
            "smile_auc_used": smile_auc,
            "unrelated_mean_auc_used": unrel_auc,
            "smile_entailed_mean_auc_used": ent_auc,
            "identity_auc_used": id_auc,
            "smile": bool(smile_auc is not None and smile_auc >= SMILE_AUC_MIN),
            "unrelated_attributes": bool(unrel_auc is not None and unrel_auc >= ATTRIBUTE_GROUP_AUC_MIN),
            "smile_entailed_attributes": bool(ent_auc is not None and ent_auc >= ATTRIBUTE_GROUP_AUC_MIN),
            "identity": None if ident["skipped"] else bool(id_auc is not None and id_auc >= IDENTITY_AUC_MIN),
            "per_attribute": per_attr_rel,
        },
    }
    out = Path(results) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluator_validation.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train and validate the attribute evaluator.")
    ap.add_argument("--data-root", default="data/celeba_64")
    ap.add_argument("--recon-run", default=None)
    ap.add_argument("--out", default="runs/evaluator")
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-identity", action="store_true")
    a = ap.parse_args(argv)
    path = train_evaluator(a.data_root, a.out, a.recon_run, a.epochs, a.subset, a.batch_size, a.lr, a.seed)
    ident = None if a.no_identity else IdentityEmbedder(pretrained=True)
    rep = validate_evaluator(path, a.data_root, a.recon_run, a.results, ident)
    print(json.dumps({"smile": rep["smile"], "reliability": {k: v for k, v in rep["reliability"].items()
                                                              if k != "per_attribute"}}, indent=2))


if __name__ == "__main__":
    main()
