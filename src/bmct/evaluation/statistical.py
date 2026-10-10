"""Statistical helpers: bootstrap CIs and DeLong test for paired AUCs."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.stats import norm

from bmct.evaluation.metrics import roc_auc


# ---------------------------------------------------------------------------
def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float] = roc_auc,
    n_replicates: int = 2000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> Tuple[float, float, float]:
    """Return (point estimate, lower, upper) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores)
    n = len(y)
    point = statistic(y, scores)
    samples = np.empty(n_replicates)
    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            samples[i] = np.nan
            continue
        samples[i] = statistic(y[idx], scores[idx])
    lo = float(np.nanquantile(samples, alpha / 2))
    hi = float(np.nanquantile(samples, 1 - alpha / 2))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
def _compute_midranks(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_test(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """DeLong's test for the difference between two AUCs.

    Returns (auc_a - auc_b, two-sided p-value).
    Implementation follows Sun & Xu (2014) O(N log N) algorithm.
    """
    y = np.asarray(y).astype(int)
    pos = y == 1
    neg = ~pos
    m, n = pos.sum(), neg.sum()

    def _structural(scores):
        X = scores[pos]
        Y = scores[neg]
        TX = _compute_midranks(X)
        TY = _compute_midranks(Y)
        TZ = _compute_midranks(np.concatenate([X, Y]))
        aucs = (TZ[:m].sum() / m - (m + 1) / 2) / n
        v01 = (TZ[:m] - TX) / n
        v10 = 1.0 - (TZ[m:] - TY) / m
        return aucs, v01, v10

    auc_a, v01a, v10a = _structural(scores_a)
    auc_b, v01b, v10b = _structural(scores_b)

    s01 = np.cov(np.stack([v01a, v01b]))
    s10 = np.cov(np.stack([v10a, v10b]))
    s = s01 / m + s10 / n
    diff = auc_a - auc_b
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = diff / np.sqrt(max(var, 1e-12))
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(diff), float(p)
