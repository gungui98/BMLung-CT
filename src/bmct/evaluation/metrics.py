"""Classification and segmentation metrics used across the paper tables."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _to_prob(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.min() >= 0.0 and scores.max() <= 1.0:
        return scores
    return _sigmoid(scores)


# ---------------------------------------------------------------------------
def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y, scores))


def brier_score(y: np.ndarray, scores: np.ndarray) -> float:
    return float(brier_score_loss(y, _to_prob(scores)))


# ---------------------------------------------------------------------------
def ece(y: np.ndarray, scores: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-mass bins."""
    p = _to_prob(scores)
    y = np.asarray(y)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    err = 0.0
    N = len(y)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        idx = (p >= lo) & (p <= hi) if i == len(edges) - 2 else (p >= lo) & (p < hi)
        if idx.sum() == 0:
            continue
        conf = p[idx].mean()
        acc = y[idx].mean()
        err += abs(conf - acc) * idx.sum() / N
    return float(err)


# ---------------------------------------------------------------------------
def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    inter = (pred * target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    """95-percentile Hausdorff distance in voxels. 2-D masks only."""
    from scipy.ndimage import distance_transform_edt
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    dt_pred = distance_transform_edt(1 - pred)
    dt_tar  = distance_transform_edt(1 - target)
    d1 = dt_tar[pred > 0]
    d2 = dt_pred[target > 0]
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


# ---------------------------------------------------------------------------
def operating_point(
    y: np.ndarray,
    scores: np.ndarray,
    target_sensitivity: float = 0.90,
) -> Tuple[float, float, float]:
    """Threshold achieving ≥ ``target_sensitivity`` with best specificity.

    Returns (threshold, sensitivity, specificity)."""
    scores = _to_prob(scores)
    thr = np.sort(np.unique(scores))[::-1]
    best = (0.5, 0.0, 0.0)
    for t in thr:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens >= target_sensitivity and spec > best[2]:
            best = (float(t), float(sens), float(spec))
    return best
