"""Pure metric functions (numpy in, numpy out). Images are in [0, 1] unless stated otherwise."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from skimage.metrics import structural_similarity

# Attributes that a smile edit should not change.
UNRELATED_ATTRIBUTES = [
    "Eyeglasses", "Male", "Young", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald",
    "Bangs", "Wearing_Hat", "Heavy_Makeup", "Pale_Skin", "No_Beard", "Mustache", "Goatee",
    "Sideburns", "Wearing_Earrings", "Wearing_Necktie", "Wearing_Lipstick", "Wavy_Hair",
    "Straight_Hair",
]
# Attributes that legitimately change when a face smiles. Reported separately, never folded into leakage.
SMILE_ENTAILED_ATTRIBUTES = ["High_Cheekbones", "Mouth_Slightly_Open", "Narrow_Eyes", "Rosy_Cheeks"]


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _flat(x) -> np.ndarray:
    a = _np(x)
    return a.reshape(a.shape[0], -1)


def recon_mse(x, y) -> np.ndarray:
    """Per-image mean squared error, [N]."""
    return ((_flat(x) - _flat(y)) ** 2).mean(axis=1)


def recon_mae(x, y) -> np.ndarray:
    """Per-image mean absolute error, [N]."""
    return np.abs(_flat(x) - _flat(y)).mean(axis=1)


def recon_ssim(x, y) -> np.ndarray:
    """Per-image SSIM for [N,3,H,W] images in [0,1] (channel axis first), [N]."""
    a, b = _np(x), _np(y)
    out = np.empty(a.shape[0])
    for i in range(a.shape[0]):
        out[i] = structural_similarity(
            np.transpose(a[i], (1, 2, 0)), np.transpose(b[i], (1, 2, 0)), channel_axis=2, data_range=1.0
        )
    return out


def monotonic_mask(scores) -> np.ndarray:
    """scores [N,L]; True where the scores strictly increase across all L levels."""
    s = _np(scores)
    return np.all(np.diff(s, axis=1) > 0, axis=1)


def monotonic_fraction(scores) -> float:
    return float(monotonic_mask(scores).mean())


def spearman_per_image(scores) -> np.ndarray:
    """Spearman correlation between level index and score for each image, [N].
    Ties are rank-averaged. If an image's scores are constant the correlation is undefined and 0.0 is returned."""
    s = _np(scores)
    L = s.shape[1]
    r = rankdata(s, axis=1)
    lv = np.arange(1, L + 1, dtype=np.float64)
    rc = r - r.mean(axis=1, keepdims=True)
    lc = lv - lv.mean()
    denom = np.sqrt((rc ** 2).sum(axis=1) * (lc ** 2).sum())
    num = (rc * lc).sum(axis=1)
    out = np.zeros(s.shape[0])
    ok = denom > 0
    out[ok] = num[ok] / denom[ok]
    return out


def score_range(scores) -> np.ndarray:
    """Per-image last-level minus first-level score, [N]."""
    s = _np(scores)
    return s[:, -1] - s[:, 0]


def monotonicity_metrics(scores) -> dict:
    s = _np(scores)
    return {
        "monotonic_fraction": monotonic_fraction(s),
        "spearman_mean": float(spearman_per_image(s).mean()),
        "score_range_mean": float(score_range(s).mean()),
    }


def cosine_similarity(a, b, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity of [N,D] arrays, [N]."""
    a, b = _flat(a), _flat(b)
    num = (a * b).sum(axis=1)
    den = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), eps)
    return num / den


def attribute_groups(attr_names) -> dict:
    """Indices into attr_names for the unrelated and smile-entailed attribute groups (names absent are skipped)."""
    names = list(attr_names)

    def idx(group):
        return [names.index(n) for n in group if n in names]

    return {"unrelated": idx(UNRELATED_ATTRIBUTES), "smile_entailed": idx(SMILE_ENTAILED_ATTRIBUTES)}


def mean_abs_change(p_a, p_b, idx) -> np.ndarray:
    """Per-image mean over the attribute indices of |p_a - p_b|; p_* are [N,40], result [N]."""
    a, b = _np(p_a), _np(p_b)
    return np.abs(a[:, idx] - b[:, idx]).mean(axis=1)
