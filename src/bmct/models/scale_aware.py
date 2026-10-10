"""Scale-aware pooling with small/medium/large experts.

Rationale: Vietnamese chest-CT presentations span a wide lesion-size
distribution, from sub-centimetre ground-glass opacities to cavitary masses
exceeding 5 cm. A single global pooling head tends to over-fit the dominant
size regime; we instead dispatch each case embedding through three
size-specialised experts and fuse them with learnt weights derived from the
lesion-map statistics.

Given an FPN pyramid, we attach one expert MLP per level:

    small  experts read P2 (finest)       → sensitive to <10 mm
    medium experts read P3               → 10–20 mm
    large  experts read P4/P5 (coarsest) → >20 mm

The dispatcher weights are a soft three-way distribution derived from the
predicted lesion map (if present) or from the slice embedding otherwise.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Expert(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_ch, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScaleAwarePool(nn.Module):
    def __init__(
        self,
        fpn_channels: int = 256,
        embed_dim: int = 256,
        size_boundaries_mm: Tuple[float, float] = (10.0, 20.0),
    ) -> None:
        super().__init__()
        self.small  = _Expert(fpn_channels, embed_dim)
        self.medium = _Expert(fpn_channels, embed_dim)
        self.large  = _Expert(fpn_channels, embed_dim)
        self.size_boundaries_mm = size_boundaries_mm
        # Dispatcher: uses a global pooled feature to produce 3-way weights
        self.dispatcher = nn.Sequential(
            nn.Linear(fpn_channels + 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        self.embed_dim = embed_dim

    # ---------------------------------------------------------------------
    def _size_stats(self, lesion_map: torch.Tensor) -> torch.Tensor:
        """Extract three summary statistics from the predicted lesion map.

        Returns a (B, 3) tensor: (mean activation, max activation,
        effective-area fraction). These feed the dispatcher so its weighting
        can adapt to the size regime at inference — no ground-truth size
        needed.
        """
        p = torch.sigmoid(lesion_map).flatten(2)    # (B, 1, H*W)
        mean = p.mean(dim=-1).squeeze(1)
        mx = p.max(dim=-1).values.squeeze(1)
        area = (p > 0.5).float().mean(dim=-1).squeeze(1)
        return torch.stack([mean, mx, area], dim=-1)    # (B, 3)

    # ---------------------------------------------------------------------
    def forward(
        self,
        pyramid: Sequence[torch.Tensor],
        bk_shape: Tuple[int, int],
        lesion_map_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (case_embed, expert_weights)

        - ``case_embed``: (B, embed_dim) fused case representation.
        - ``expert_weights``: (B, 3) soft dispatcher probabilities.
        """
        B, K = bk_shape
        # Pool slice dim: average over K (post-MIL this fuses per-case)
        # We use P2 (finest) for small, P3 for medium, P4/P5 avg for large.
        p2, p3, p4, p5 = pyramid[0], pyramid[1], pyramid[2], pyramid[3]

        def _bk_mean(feat: torch.Tensor) -> torch.Tensor:
            # feat: (B*K, C, H, W) → (B, C, H, W)
            C, H, W = feat.shape[1:]
            return feat.view(B, K, C, H, W).mean(dim=1)

        small_in  = _bk_mean(p2)
        medium_in = _bk_mean(p3)
        p4_m = _bk_mean(p4)
        p5_m = _bk_mean(p5)
        # FPN levels have different spatial extents; resample P5 to match P4
        # before averaging the "large" expert input.
        if p5_m.shape[-2:] != p4_m.shape[-2:]:
            p5_m = F.interpolate(p5_m, size=p4_m.shape[-2:],
                                  mode="bilinear", align_corners=False)
        large_in  = 0.5 * (p4_m + p5_m)

        s_emb = self.small(small_in)
        m_emb = self.medium(medium_in)
        l_emb = self.large(large_in)

        # Dispatcher input = global avg of P3 plus lesion-map summaries
        pooled = F.adaptive_avg_pool2d(medium_in, 1).flatten(1)
        stats = self._size_stats(lesion_map_logits)
        gate_in = torch.cat([pooled, stats], dim=-1)
        weights = torch.softmax(self.dispatcher(gate_in), dim=-1)   # (B, 3)

        fused = (weights[:, 0:1] * s_emb
                 + weights[:, 1:2] * m_emb
                 + weights[:, 2:3] * l_emb)
        return fused, weights
