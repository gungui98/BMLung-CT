"""Tversky loss — generalised Dice with α/β controlling FP vs FN penalty.

    Tversky = TP / (TP + α·FP + β·FN)
    Loss    = 1 - Tversky

For lung-lesion segmentation we typically pick β > α so the model is
penalised more for missing lesion pixels than for over-segmenting.
"""
from __future__ import annotations

import torch


def soft_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """logits / target: (B, 1, H, W). Returns scalar."""
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum(dim=(1, 2, 3))
    fp = (probs * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - probs) * target).sum(dim=(1, 2, 3))
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - tv).mean()
