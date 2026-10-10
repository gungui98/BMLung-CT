"""The proposed end-to-end multimodal model.

Architecture (see manuscript Figure 2):
    2.5-D ResNet-50 + FPN encoder
        ├── Lesion-map head        → (B, 1, Y, X) logits    L_seg
        ├── Scale-aware pool       → (B, D) with (B, 3) expert weights
        └── Gated-attention MIL    → (B, D) bag embed, (B, K) attention
    Concatenated image embedding is fed into:
        ├── Case classifier         → malignancy logit        L_case
        ├── Property heads          → one logit vector/group  L_prop
        └── Consistency hinge with lesion-map attention       L_cons

Clinical-branch features are late-fused with the image case embedding
before the malignancy head.

Outputs are returned as a dict so the training loop can slice cleanly.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from bmct.models.clinical_branch import ClinicalBranch
from bmct.models.encoder import Encoder25D
from bmct.models.lesion_map import LesionMapHead
from bmct.models.mil import GatedAttentionMIL
from bmct.models.property_heads import PropertyHeads
from bmct.models.scale_aware import ScaleAwarePool


class ScaleAwareE2E(nn.Module):
    """``cfg`` is the ``params:`` block of ``configs/model/scale_aware_e2e.yaml``."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        fpn = int(cfg.fpn_channels)
        in_channels = int(cfg.in_channels)
        self.encoder = Encoder25D(in_channels=in_channels,
                                   pretrained=bool(cfg.get("pretrained", True)),
                                   fpn_channels=fpn)
        embed = int(cfg.get("fusion", {}).get("hidden", 128))

        self.lesion_head = LesionMapHead(
            in_channels=fpn,
            hidden=int(cfg.lesion_head.hidden),
        )
        self.scale_pool = ScaleAwarePool(
            fpn_channels=fpn,
            embed_dim=embed,
            size_boundaries_mm=tuple(cfg.scale_aware_pool.size_boundaries_mm),
        )
        self.mil = GatedAttentionMIL(
            embed_dim=self.encoder.embed_dim,
            hidden_dim=int(cfg.mil.attention_dim),
        )
        self.prop_heads = PropertyHeads(
            embed_dim=embed + self.encoder.embed_dim,
            n_classes=dict(cfg.property_heads),
        )
        self.clinical = ClinicalBranch(
            in_features=int(cfg.get("clinical_in_features", 8)),
            out_features=int(cfg.clinical_branch.hidden),
            hidden=int(cfg.clinical_branch.hidden),
            dropout=float(cfg.clinical_branch.dropout),
        )
        fusion_dim = embed + self.encoder.embed_dim + self.clinical.out_features
        self.case_head = nn.Sequential(
            nn.Linear(fusion_dim, int(cfg.fusion.hidden)),
            nn.GELU(),
            nn.Dropout(float(cfg.fusion.dropout)),
            nn.Linear(int(cfg.fusion.hidden), 1),
        )

    # ---------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,           # (B, K, C, Y, X)
        clinical: torch.Tensor,        # (B, n_clinical)
    ) -> Dict[str, torch.Tensor]:
        enc = self.encoder(image)
        pyramid = enc["pyramid"]
        bk_shape = enc["shape"]

        lesion_logits = self.lesion_head(
            pyramid[0], bk_shape, upsample_to=image.shape[-1]
        )                                                      # (B, 1, Y, X)

        fused_img, expert_w = self.scale_pool(pyramid, bk_shape, lesion_logits)
        bag_embed, attn = self.mil(enc["slice_embed"])          # (B, D), (B, K)

        image_embed = torch.cat([fused_img, bag_embed], dim=-1)
        prop_logits = self.prop_heads(image_embed)

        c = self.clinical(clinical)
        case_logit = self.case_head(torch.cat([image_embed, c], dim=-1)).squeeze(-1)

        return {
            "case_logit":   case_logit,        # (B,)
            "bag_logit":    case_logit,        # re-used for optional L_bag
            "seg_logits":   lesion_logits,     # (B, 1, Y, X)
            "prop_logits":  prop_logits,       # dict[group] → (B, n_g)
            "attention":    attn,              # (B, K)
            "expert_w":     expert_w,          # (B, 3)
        }
