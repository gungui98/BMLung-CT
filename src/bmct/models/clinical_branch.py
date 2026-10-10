"""Dense clinical branch for late fusion."""
from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalBranch(nn.Module):
    def __init__(self, in_features: int, out_features: int = 64,
                 hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
            nn.GELU(),
        )
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
