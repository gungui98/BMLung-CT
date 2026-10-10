"""Soft lesion-map head.

Given the finest-resolution FPN feature map, predict a single-channel logit
map of lesion presence per slice.  The map is both:
  * supervised (when M_seg = 1) by the Dice + BCE loss in
    ``bmct.losses.masked_multitask`` (see ``losses/dice.py``), and
  * consumed as an attention prior by the gated-attention MIL head through
    the ``consistency`` term in the loss (keeps the MIL attention aligned with
    the lesion map on annotated cases).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LesionMapHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(
        self,
        feat_bk_chw: torch.Tensor,
        bk_shape,
        upsample_to: int | None = None,
    ) -> torch.Tensor:
        """Return (B, 1, Y, X) — case-level lesion map, max-pooled over slices.

        ``feat_bk_chw`` is (B*K, C, H, W). ``bk_shape`` is (B, K).  We compute
        logits per slice, up-sample to the target resolution and aggregate
        across slices by max — matching the supervised target which is a
        single 2-D projection of the lesion mask.
        """
        B, K = bk_shape
        logits = self.net(feat_bk_chw)                      # (B*K, 1, H, W)
        if upsample_to is not None:
            logits = F.interpolate(logits, size=upsample_to,
                                    mode="bilinear", align_corners=False)
        logits = logits.view(B, K, 1, *logits.shape[-2:])
        return logits.max(dim=1).values                    # (B, 1, Y, X)
