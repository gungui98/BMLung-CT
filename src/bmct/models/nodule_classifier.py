"""Per-nodule classifier (stage 2 of the seg → classify pipeline).

Backbone: torchvision ResNet18 with the first conv adapted to a configurable
number of input channels (3 by default — three adjacent axial slices stacked
into the channel dim, matching :class:`NoduleDataset` output).

Heads (one linear off the global-pool feature each):
    density (3-way), margin (4-way) — softmax + CE
    cavitation (binary), cancer (binary) — sigmoid + BCE
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
import torchvision


def _adapt_first_conv(old: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    new = nn.Conv2d(in_channels, old.out_channels,
                     kernel_size=old.kernel_size, stride=old.stride,
                     padding=old.padding, bias=old.bias is not None)
    if pretrained:
        with torch.no_grad():
            w = old.weight
            avg = w.mean(dim=1, keepdim=True)
            new.weight.copy_(avg.repeat(1, in_channels, 1, 1)
                             * (w.shape[1] / in_channels))
    return new


def _make_backbone(in_channels: int, pretrained: bool,
                    name: str = "resnet18") -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        if in_channels != 3:
            net.conv1 = _adapt_first_conv(net.conv1, in_channels, pretrained)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        if in_channels != 3:
            net.conv1 = _adapt_first_conv(net.conv1, in_channels, pretrained)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "densenet121":
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.densenet121(weights=weights)
        if in_channels != 3:
            net.features.conv0 = _adapt_first_conv(net.features.conv0,
                                                    in_channels, pretrained)
        feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
    elif name in {"convnext_tiny", "swin_tiny_patch4_window7_224"}:
        import timm
        net = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                  in_chans=in_channels)
        feat_dim = net.num_features
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return net, feat_dim


class NoduleClassifier(nn.Module):
    """``cfg`` (model YAML ``params:``):
        in_channels: int (3 = three adjacent slices)
        pretrained:  bool
        dropout:     float
        n_density: 3
        n_margin:  4
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        in_channels = int(cfg.get("in_channels", 3))
        pretrained = bool(cfg.get("pretrained", True))
        dropout = float(cfg.get("dropout", 0.2))
        n_density = int(cfg.get("n_density", 3))
        n_margin = int(cfg.get("n_margin", 4))

        backbone = str(cfg.get("backbone", "resnet18"))
        self.backbone, d = _make_backbone(in_channels, pretrained, backbone)
        self.drop = nn.Dropout(dropout)
        self.head_density = nn.Linear(d, n_density)
        self.head_margin  = nn.Linear(d, n_margin)
        self.head_cavit   = nn.Linear(d, 1)
        self.head_cancer  = nn.Linear(d, 1)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.drop(self.backbone(image))
        return {
            "density": self.head_density(f),       # (B, n_density)
            "margin":  self.head_margin(f),        # (B, n_margin)
            "cavit":   self.head_cavit(f).squeeze(-1),    # (B,)
            "cancer":  self.head_cancer(f).squeeze(-1),   # (B,)
        }
