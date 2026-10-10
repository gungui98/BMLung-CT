"""Plain 2-D U-Net segmenter for the slice-level dataset.

Input is the per-slice tensor ``(B, C, H, W)`` from :class:`BMSliceDataset`
— no key-slice / 2.5-D folding, no MIL pooling. Output is the raw
seg logits with the same spatial dims.

This is the segmentation sanity-check model. Once it's clear the lesion
signal is learnable per-slice, we add the case/property/clinical heads.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig


class UNet2DSlice(nn.Module):
    """3-level 2-D U-Net.

    ``cfg`` (model YAML ``params:``):
        in_channels:  int  (1 for visible-window single-channel input)
        base_channels: int (16 = small / 32 = standard)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        base = int(cfg.get("base_channels", 16))
        in_channels = int(cfg.get("in_channels", 1))

        self.enc1 = self._block(in_channels, base)
        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.dec2 = self._block(base * 4 + base * 2, base * 2)
        self.dec1 = self._block(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=False)

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        seg_logits = self.out(d1)                          # (B, 1, H, W)
        return {"seg_logits": seg_logits}
