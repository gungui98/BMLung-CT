"""Gated-attention multiple-instance learning (Ilse et al., 2018)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.V = nn.Linear(embed_dim, hidden_dim)
        self.U = nn.Linear(embed_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, instances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``instances`` shape (B, K, D).

        Returns (bag_embed [B,D], attention_weights [B,K])."""
        v = torch.tanh(self.V(instances))          # (B, K, H)
        u = torch.sigmoid(self.U(instances))       # (B, K, H)
        scores = self.w(v * u).squeeze(-1)         # (B, K)
        attn = F.softmax(scores, dim=-1)           # (B, K)
        bag = torch.einsum("bk,bkd->bd", attn, instances)
        return bag, attn
