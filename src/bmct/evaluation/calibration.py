"""Calibration utilities: reliability curve + temperature scaling + Brier."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def brier_score(y: np.ndarray, scores: np.ndarray) -> float:
    """Binary Brier score: mean((p - y)^2). Lower = better calibrated.

    ``y`` ∈ {0,1} (or float); ``scores`` ∈ [0,1] (probabilities, not logits).
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y={y.shape} scores={p.shape}")
    return float(np.mean((p - y) ** 2))


def brier_score_multiclass(y: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier: mean over samples of sum_k (p_k - 1{y=k})^2.

    ``y`` ∈ {0..K-1}, shape (N,);  ``probs`` shape (N, K), rows sum to 1.
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    n, k = p.shape
    if y.shape != (n,):
        raise ValueError(f"shape mismatch: y={y.shape} probs={p.shape}")
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def expected_calibration_error(
    y: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> float:
    """ECE: weighted L1 gap between confidence and accuracy across bins."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    n = len(p)
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        idx = (p >= lo) & (p <= hi) if i == len(edges) - 2 else (p >= lo) & (p < hi)
        if idx.sum() == 0:
            continue
        ece += (idx.sum() / n) * abs(p[idx].mean() - y[idx].mean())
    return float(ece)


def calibration_curve(
    y: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_predicted, mean_observed) for each bin."""
    p = np.clip(scores, 0.0, 1.0)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    mean_pred, mean_obs = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        idx = (p >= lo) & (p <= hi) if i == len(edges) - 2 else (p >= lo) & (p < hi)
        if idx.sum() == 0:
            continue
        mean_pred.append(p[idx].mean())
        mean_obs.append(y[idx].mean())
    return np.asarray(mean_pred), np.asarray(mean_obs)


def temperature_scale(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Fit a scalar temperature T minimising NLL on held-out logits."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(_closure)
    return float(T.detach().clamp_min(1e-3).item())
