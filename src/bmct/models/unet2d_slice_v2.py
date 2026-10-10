"""4-level 2-D U-Net with optional 2.5-D input (channel dim = stack_size).

Identical encoder/decoder pattern as :class:`UNet2DSlice` but one level deeper
(downsample 8× instead of 4×), so receptive field doubles. base=48 keeps the
parameter count around 6-7M with 4 levels.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig


class UNet2DSliceV2(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        base = int(cfg.get("base_channels", 48))
        in_channels = int(cfg.get("in_channels", 3))

        self.enc1 = self._block(in_channels, base)
        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.enc4 = self._block(base * 4, base * 8)
        self.dec3 = self._block(base * 8 + base * 4, base * 4)
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
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return {"seg_logits": self.out(d1)}
