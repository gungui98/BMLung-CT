"""Per-property classification heads.

One head per property group declared in ``bmct.data.schema.PROPERTY_GROUPS``.
Each head consumes the fused case embedding and predicts a categorical
distribution over the group's classes.  Training masks unannotated cases via
the M_prop indicator in the loss.
"""
from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


class PropertyHeads(nn.Module):
    def __init__(self, embed_dim: int, n_classes: Mapping[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict({
            group: nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, n),
            )
            for group, n in n_classes.items()
        })

    def forward(self, embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {g: self.heads[g](embed) for g in self.heads}
