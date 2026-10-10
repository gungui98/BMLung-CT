"""Thin wrappers around `segmentation_models_pytorch` so the U-Net++ and
DeepLabV3+ baselines plug into the existing slice-level training pipeline.

Both wrappers expose a single ``image -> {"seg_logits": (B, 1, H, W)}``
forward to match :class:`UNet2DSlice`. Backbone is configurable through the
model YAML ``params.encoder_name`` field (default ``resnet34``).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

import segmentation_models_pytorch as smp


def _build(arch_cls, cfg: DictConfig) -> nn.Module:
    return arch_cls(
        encoder_name=str(cfg.get("encoder_name", "resnet34")),
        encoder_weights=cfg.get("encoder_weights", "imagenet")
            if cfg.get("encoder_weights", "imagenet") not in (None, "none", "None")
            else None,
        in_channels=int(cfg.get("in_channels", 1)),
        classes=1,
    )


class UNetPlusPlus2D(nn.Module):
    """U-Net++ (nested skip connections) wrapper, 2-D."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.net = _build(smp.UnetPlusPlus, cfg)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"seg_logits": self.net(image)}


class DeepLabV3Plus2D(nn.Module):
    """DeepLabV3+ (ASPP multi-scale) wrapper, 2-D."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.net = _build(smp.DeepLabV3Plus, cfg)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"seg_logits": self.net(image)}
