"""2.5-D ResNet+FPN encoder shared by every image branch.

The encoder consumes a (B, K, C, Y, X) tensor where ``K`` is the number of
key slices sampled per case. The K dimension is folded into the batch,
processed slice-wise, then unfolded so the downstream pooling sees a
(B, K, D) embedding for gated-attention MIL, or a feature map stack for the
lesion-map head and scale-aware pool.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ---------------------------------------------------------------------------
class _FPN(nn.Module):
    """Top-down feature pyramid with lateral 1x1 convs."""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])
        self.smooths = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # ``feats`` is shallow→deep (P2,P3,P4,P5 with 256→2048 channels).
        # Walk from deepest to shallowest so each level is fed through the
        # lateral conv that matches its own channel count.
        outs: List[torch.Tensor] = [None] * len(feats)
        prev = self.laterals[-1](feats[-1])
        outs[-1] = self.smooths[-1](prev)
        for i in range(len(feats) - 2, -1, -1):
            lat = self.laterals[i](feats[i])
            up = F.interpolate(prev, size=lat.shape[-2:], mode="nearest")
            prev = lat + up
            outs[i] = self.smooths[i](prev)
        return outs  # shallow→deep order


# ---------------------------------------------------------------------------
class Encoder25D(nn.Module):
    """ResNet-50 backbone + FPN operating slice-wise on 2.5-D stacks."""

    def __init__(
        self,
        in_channels: int = 6,
        pretrained: bool = True,
        fpn_channels: int = 256,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        # adapt first conv to accept ``in_channels`` (e.g. 2 windows × 3 slices)
        if in_channels != 3:
            w = backbone.conv1.weight.data
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7,
                                  stride=2, padding=3, bias=False)
            with torch.no_grad():
                rep = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                new_conv.weight.copy_(rep / in_channels * 3)
            backbone.conv1 = new_conv

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1          # 256
        self.layer2 = backbone.layer2          # 512
        self.layer3 = backbone.layer3          # 1024
        self.layer4 = backbone.layer4          # 2048

        self.fpn = _FPN([256, 512, 1024, 2048], out_channels=fpn_channels)
        self.out_channels = fpn_channels

        # slice-level embedding used by gated attention MIL
        self.slice_pool = nn.AdaptiveAvgPool2d(1)
        self.embed_dim = fpn_channels

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return a dict with:
            ``pyramid``: list of (B*K, fpn, H_l, W_l) tensors, shallow→deep
            ``slice_embed``: (B, K, fpn)
        """
        B, K = x.shape[:2]
        x = x.view(B * K, *x.shape[2:])
        s = self.stem(x)
        c2 = self.layer1(s)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        pyramid = self.fpn([c2, c3, c4, c5])
        slice_embed = self.slice_pool(pyramid[0]).flatten(1).view(B, K, -1)
        return {
            "pyramid": pyramid,                # list, each (B*K, fpn, H, W)
            "slice_embed": slice_embed,        # (B, K, D)
            "shape": (B, K),
        }


# ---------------------------------------------------------------------------
def build_encoder(cfg) -> Encoder25D:
    return Encoder25D(
        in_channels=cfg.get("in_channels", 6),
        pretrained=cfg.get("pretrained", True),
        fpn_channels=cfg.get("fpn_channels", 256),
    )
