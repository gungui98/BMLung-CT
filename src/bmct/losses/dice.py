"""Soft Dice loss with optional per-sample mask."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """logits and target are (B, 1, Y, X). ``mask`` is (B,) 0/1."""
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    num = 2 * (probs * target).sum(dim=dims)
    den = probs.sum(dim=dims) + target.sum(dim=dims) + eps
    dice = 1.0 - num / den                                # (B,)
    if mask is not None:
        dice = dice * mask
        denom = mask.sum().clamp_min(1.0)
        return dice.sum() / denom
    return dice.mean()
