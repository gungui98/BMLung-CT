"""Model registry keyed by the ``target`` string declared in each config.

Every ``configs/model/*.yaml`` file has the shape

    name: <short-name>
    target: bmct.models....
    params:
        <nested hyperparameters>

``build_model(cfg)`` reads ``cfg.target`` and constructs the referenced class
with ``cfg.params`` — so every model class receives a single flat
OmegaConf ``DictConfig`` of its own hyperparameters.
"""
from __future__ import annotations

from typing import Any, Dict, Type

from omegaconf import DictConfig

from bmct.models.baselines import (
    ClinicalOnly,
    ImageOnlyMIL,
    RadiomicsClinical,
    SegReference,
    UNet2D,
)
from bmct.models.scale_aware_e2e import ScaleAwareE2E
from bmct.models.unet2d_slice import UNet2DSlice
from bmct.models.unet2d_slice_v2 import UNet2DSliceV2
from bmct.models.smp_wrappers import UNetPlusPlus2D, DeepLabV3Plus2D


MODEL_REGISTRY: Dict[str, Type] = {
    "bmct.models.scale_aware_e2e.ScaleAwareE2E":     ScaleAwareE2E,
    "bmct.models.baselines.ClinicalOnly":            ClinicalOnly,
    "bmct.models.baselines.RadiomicsClinical":       RadiomicsClinical,
    "bmct.models.baselines.ImageOnlyMIL":            ImageOnlyMIL,
    "bmct.models.baselines.SegReference":            SegReference,
    "bmct.models.baselines.UNet2D":                  UNet2D,
    "bmct.models.unet2d_slice.UNet2DSlice":          UNet2DSlice,
    "bmct.models.unet2d_slice_v2.UNet2DSliceV2":     UNet2DSliceV2,
    "bmct.models.smp_wrappers.UNetPlusPlus2D":       UNetPlusPlus2D,
    "bmct.models.smp_wrappers.DeepLabV3Plus2D":      DeepLabV3Plus2D,
}


def build_model(cfg: DictConfig) -> Any:
    # accept both {target, params} (paper-style) and {_target_} (flat)
    target = cfg.get("target") or cfg.get("_target_")
    if target not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model target '{target}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    params = cfg.get("params", cfg)
    return MODEL_REGISTRY[target](params)
