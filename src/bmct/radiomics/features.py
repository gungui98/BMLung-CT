"""Thin wrapper around pyradiomics.featureextractor.

Defaults to the "all" feature class set with standard resampling (1×1×1 mm).
Writes a parquet/csv keyed by case_id so the RadiomicsClinical baseline can
do a simple left-join against the clinical table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from radiomics import featureextractor
    _PYR_OK = True
except ImportError:
    _PYR_OK = False


_DEFAULT_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": [1.0, 1.0, 1.0],
    "interpolator": "sitkBSpline",
    "normalize": True,
    "normalizeScale": 100,
    "voxelArrayShift": 1000,
}
_DEFAULT_CLASSES = ["firstorder", "shape", "glcm", "glrlm", "glszm", "ngtdm", "gldm"]


def _build_extractor(custom_settings: Optional[dict] = None):
    if not _PYR_OK:
        raise ImportError("Install with `pip install bmct[radiomics]`.")
    settings = dict(_DEFAULT_SETTINGS)
    if custom_settings:
        settings.update(custom_settings)
    ex = featureextractor.RadiomicsFeatureExtractor(**settings)
    ex.disableAllFeatures()
    for c in _DEFAULT_CLASSES:
        ex.enableFeatureClassByName(c)
    return ex


def extract_features(image_path: str, mask_path: str,
                     settings: Optional[dict] = None) -> Dict[str, float]:
    ex = _build_extractor(settings)
    out = ex.execute(image_path, mask_path)
    # Drop diagnostics keys (prefixed 'diagnostics_'); pyradiomics returns
    # 0-d numpy arrays which np.isscalar() rejects — accept anything with
    # ndim == 0 as a scalar.
    feats: Dict[str, float] = {}
    for k, v in out.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            if np.ndim(v) == 0:
                feats[k] = float(v)
        except Exception:
            continue
    return feats


def extract_batch(
    cases: Iterable[dict],
    output_path: str,
    settings: Optional[dict] = None,
) -> pd.DataFrame:
    """``cases`` is an iterable of dicts with keys ``case_id``, ``image``,
    ``mask``. Writes a CSV at ``output_path``."""
    rows: List[dict] = []
    for c in tqdm(list(cases), desc="radiomics"):
        feats = extract_features(c["image"], c["mask"], settings)
        rows.append({"case_id": c["case_id"], **feats})
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
