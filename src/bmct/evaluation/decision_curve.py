"""Decision Curve Analysis (Vickers & Elkin, 2006).

Net benefit at threshold probability p_t is

    NB(p_t) = TP/N - FP/N * ( p_t / (1 - p_t) )

The reference curves are:
  * 'treat all':    p - (1-p) * p_t / (1-p_t)
  * 'treat none':   0
Given a range of thresholds the function returns a dataframe with one row per
(threshold, strategy) tuple — convenient for plotting.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def decision_curve(
    y: np.ndarray,
    scores: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.clip(scores, 0.0, 1.0)
    thresholds = np.asarray(
        list(thresholds) if thresholds is not None
        else np.linspace(0.01, 0.99, 99)
    )
    N = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        w = t / max(1 - t, 1e-9)
        nb_model = tp / N - fp / N * w
        nb_all = prev - (1 - prev) * w
        rows.extend([
            {"threshold": t, "strategy": "model", "net_benefit": nb_model},
            {"threshold": t, "strategy": "treat_all", "net_benefit": nb_all},
            {"threshold": t, "strategy": "treat_none", "net_benefit": 0.0},
        ])
    return pd.DataFrame(rows)
