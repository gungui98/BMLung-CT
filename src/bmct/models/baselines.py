"""Baseline models.

Four comparators, each chosen to isolate a confounder in the paper's story:

    ClinicalOnly       pure tabular baseline (age/sex/smoking/symptom/pack-years)
    RadiomicsClinical  pyradiomics features + clinical (sklearn-style)
    ImageOnlyMIL       2.5-D ResNet + gated attention MIL (no lesion-map,
                       no scale-aware pool, no property heads)
    SegReference       3-D nnU-Net-style segmentation reference:  train on the
                       200 annotated cases, infer on all cases, feed derived
                       features (max prob, area) + clinical into a small MLP.

Every constructor takes the ``params:`` block of
``configs/model/<name>.yaml``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from bmct.models.encoder import Encoder25D
from bmct.models.mil import GatedAttentionMIL


# ---------------------------------------------------------------------------
# Tabular baselines
# ---------------------------------------------------------------------------
class _SklearnBaseline(nn.Module):
    """Wrap a scikit-learn estimator so training uses the same CLI."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._model = None
        self._fit = False

    def _build(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        kind = self.cfg.get("classifier", "logistic")
        if kind in ("logistic", "logistic_regression"):
            clf = LogisticRegression(
                C=self.cfg.get("C", 1.0),
                max_iter=self.cfg.get("max_iter", 1000),
                class_weight="balanced",
            )
        elif kind == "gradient_boosting":
            clf = GradientBoostingClassifier(
                n_estimators=self.cfg.get("n_estimators", 300),
                learning_rate=self.cfg.get("learning_rate", 0.05),
                max_depth=self.cfg.get("max_depth", 3),
            )
        else:
            raise ValueError(f"Unknown classifier '{kind}'")
        self._model = Pipeline([("scale", StandardScaler()), ("clf", clf)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._model is None:
            self._build()
        self._model.fit(X, y)
        self._fit = True

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fit:
            raise RuntimeError("Baseline must be fit() before predict_proba().")
        return self._model.predict_proba(X)[:, 1]


class ClinicalOnly(_SklearnBaseline):
    """Clinical-variable-only logistic regression / GBM.

    Config schema (see ``configs/model/clinical.yaml``)::

        hidden_dim: 64                # unused by sklearn path
        n_hidden_layers: 2
        dropout: 0.2
        use_missing_indicators: true
        classifier: logistic          # optional override
    """


class RadiomicsClinical(_SklearnBaseline):
    """Radiomics (pyradiomics) + clinical baseline.

    Feature extraction is performed offline by
    ``scripts/prepare_manifests.py --extract-radiomics``; the training script
    concatenates radiomics and clinical before ``.fit()``.
    """


# ---------------------------------------------------------------------------
# Image-only gated-attention MIL baseline
# ---------------------------------------------------------------------------
class ImageOnlyMIL(nn.Module):
    """``cfg`` is the ``params:`` block of ``configs/model/image_mil.yaml``.

    Expected keys::

        backbone: resnet50
        in_channels: 6
        feature_dim: 2048         # informational; used internally by backbone
        attention_dim: 256
        dropout: 0.3
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.encoder = Encoder25D(
            in_channels=int(cfg.in_channels),
            pretrained=bool(cfg.get("pretrained", True)),
            fpn_channels=int(cfg.get("fpn_channels", 256)),
        )
        self.mil = GatedAttentionMIL(
            embed_dim=self.encoder.embed_dim,
            hidden_dim=int(cfg.attention_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, 128),
            nn.GELU(),
            nn.Dropout(float(cfg.get("dropout", 0.3))),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor,
                clinical: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        enc = self.encoder(image)
        bag, attn = self.mil(enc["slice_embed"])
        logit = self.head(bag).squeeze(-1)
        return {"case_logit": logit, "attention": attn}


# ---------------------------------------------------------------------------
# 3-D segmentation reference
# ---------------------------------------------------------------------------
class SegReference(nn.Module):
    """Small 3-D U-Net-style segmenter + downstream MLP over derived features.

    ``cfg`` is the ``params:`` block of ``configs/model/seg_reference.yaml``:

        base_channels: 32
        depth: 5
        dropout: 0.1
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        base = int(cfg.get("base_channels", 32))
        in_channels = int(cfg.get("in_channels", 2))
        self.enc1 = self._block(in_channels, base)
        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.dec2 = self._block(base * 4 + base * 2, base * 2)
        self.dec1 = self._block(base * 2 + base, base)
        self.out = nn.Conv3d(base, 1, kernel_size=1)
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear",
                                     align_corners=False)
        self.mlp = nn.Sequential(
            nn.Linear(4 + int(cfg.get("clinical_in_features", 8)), 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv3d(cout, cout, kernel_size=3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def _unet(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.upsample(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        return self.out(d1)

    def forward(self, volume: torch.Tensor,
                clinical: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``volume`` is (B, C, Z, Y, X). Returns case_logit + seg_logits."""
        seg = self._unet(volume)                       # (B, 1, Z, Y, X)
        p = torch.sigmoid(seg)
        derived = torch.stack([
            p.amax(dim=(1, 2, 3, 4)),
            p.mean(dim=(1, 2, 3, 4)),
            (p > 0.5).float().mean(dim=(1, 2, 3, 4)),
            p.flatten(1).topk(64, dim=1).values.mean(dim=1),
        ], dim=-1)                                     # (B, 4)
        logit = self.mlp(torch.cat([derived, clinical], dim=-1)).squeeze(-1)
        return {"case_logit": logit, "seg_logits": seg}


# ---------------------------------------------------------------------------
# Plain 2-D U-Net segmentation baseline (consumes the 2.5-D loader output)
# ---------------------------------------------------------------------------
class UNet2D(nn.Module):
    """Per-slice 2-D U-Net + max-projection over the key-slice axis.

    Input shape matches the rest of the pipeline: ``(B, K, C, Y, X)``.
    The K slices are flattened into the batch dimension, a standard 2-D
    U-Net predicts a per-slice mask, and the max over K produces the 2-D
    mask the existing seg loss expects (it max-projects the GT too).
    Case-level logit is the global max of the predicted sigmoid map —
    so a high-confidence lesion pixel anywhere implies high cancer prob.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        base = int(cfg.get("base_channels", 16))
        in_channels = int(cfg.get("in_channels", 6))
        # 3-level U-Net keeps activations small enough for (B*K, C, H, W)
        # folding at batch=16, K=16, base=16, H=W=352 to fit on a 48 GB GPU.
        self.enc1 = self._block(in_channels, base)
        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.dec2 = self._block(base * 4 + base * 2, base * 2)
        self.dec1 = self._block(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=False)

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor,
                clinical: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``image`` is (B, K, C, Y, X). Returns case_logit + seg_logits."""
        B, K, C, H, W = image.shape
        x = image.reshape(B * K, C, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        seg_per_slice = self.out(d1)                       # (B*K, 1, H, W)
        seg_per_slice = seg_per_slice.reshape(B, K, 1, H, W)
        seg_logits = seg_per_slice.amax(dim=1)             # (B, 1, H, W) — max over K
        case_logit = seg_logits.flatten(1).amax(dim=1)     # (B,)
        return {"case_logit": case_logit, "seg_logits": seg_logits}
