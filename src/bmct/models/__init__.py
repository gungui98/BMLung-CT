"""Model zoo for BMCT.

The registry exposes every model class behind a string key so the training
loop can instantiate them from the OmegaConf ``_target_`` field declared in
``configs/model/*.yaml``.
"""
from bmct.models.encoder import Encoder25D, build_encoder
from bmct.models.lesion_map import LesionMapHead
from bmct.models.scale_aware import ScaleAwarePool
from bmct.models.mil import GatedAttentionMIL
from bmct.models.property_heads import PropertyHeads
from bmct.models.clinical_branch import ClinicalBranch
from bmct.models.scale_aware_e2e import ScaleAwareE2E
from bmct.models.baselines import (
    ClinicalOnly,
    RadiomicsClinical,
    ImageOnlyMIL,
    SegReference,
)
from bmct.models.registry import MODEL_REGISTRY, build_model

__all__ = [
    "Encoder25D",
    "build_encoder",
    "LesionMapHead",
    "ScaleAwarePool",
    "GatedAttentionMIL",
    "PropertyHeads",
    "ClinicalBranch",
    "ScaleAwareE2E",
    "ClinicalOnly",
    "RadiomicsClinical",
    "ImageOnlyMIL",
    "SegReference",
    "MODEL_REGISTRY",
    "build_model",
]
