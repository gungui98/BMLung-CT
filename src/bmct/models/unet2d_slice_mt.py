"""Multi-task 2-D U-Net: per-slice segmentation + multi-label attribute
classification.

Same encoder/decoder as :class:`UNet2DSlice` for the seg map; in addition a
small classification branch taps the bottleneck feature ``e3`` via global
average pooling and emits one sigmoid logit per attribute class (multi-label,
not softmax — a nodule can be e.g. both ``lobulated`` and ``spiculated``).

The attribute classifier outputs a single K_total flat vector. Callers split
into groups via :data:`bmct.data.slice_dataset.GROUP_OFFSETS`.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from bmct.data.slice_dataset import K_TOTAL


class UNet2DSliceMT(nn.Module):
    """``cfg`` (model YAML ``params:``):
        in_channels:   int (1)
        base_channels: int (32)
        cls_hidden:    int (128) — MLP hidden dim of the attribute head
        cls_dropout:   float (0.2)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        base = int(cfg.get("base_channels", 32))
        in_channels = int(cfg.get("in_channels", 1))
        cls_hidden = int(cfg.get("cls_hidden", 128))
        cls_dropout = float(cfg.get("cls_dropout", 0.2))

        self.enc1 = self._block(in_channels, base)
        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.dec2 = self._block(base * 4 + base * 2, base * 2)
        self.dec1 = self._block(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=False)

        # Classification head off the bottleneck.
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, cls_hidden),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, K_TOTAL),
        )

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
        attr_logits = self.cls(e3)                          # (B, K_total)
        return {"seg_logits": seg_logits, "attr_logits": attr_logits}
