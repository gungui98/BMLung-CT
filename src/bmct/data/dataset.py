"""PyTorch Dataset and collate utilities with masked multitask supervision.

Every case contributes to the case-level (malignancy) loss.  Only cases whose
``case_id`` appears in the lesion table (``lesions.csv``) contribute to the
segmentation and property-prediction losses; the indicator masks ``M_seg`` and
``M_prop`` zero out the unsupervised terms.  See
``losses/masked_multitask.py`` for the combined loss.

The dataset derives strong-annotation membership from the lesion table — no
separate boolean column on ``CaseRecord`` is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from bmct.data.preprocessing import PreprocConfig, preprocess_case
from bmct.data.schema import (
    PROPERTY_GROUPS,
    CaseRecord,
    LesionRecord,
    load_lesion_table,
)
from skimage.transform import resize as _sk_resize


# ---------------------------------------------------------------------------
@dataclass
class MultitaskBatch:
    """Strong-typed container for one training minibatch."""
    image: torch.Tensor                        # (B, K, C, Y, X)
    case_label: torch.Tensor                   # (B,) long 0/1
    clinical: torch.Tensor                     # (B, n_clinical)
    seg_target: torch.Tensor                   # (B, 1, Y, X)  lesion-map target
    seg_mask: torch.Tensor                     # (B,) 0/1 M_seg
    prop_target: Dict[str, torch.Tensor]       # group -> (B,) long index
    prop_mask: torch.Tensor                    # (B,) 0/1 M_prop
    case_id: List[str]                         # for logging/eval

    def to(self, device) -> "MultitaskBatch":
        return MultitaskBatch(
            image=self.image.to(device, non_blocking=True),
            case_label=self.case_label.to(device, non_blocking=True),
            clinical=self.clinical.to(device, non_blocking=True),
            seg_target=self.seg_target.to(device, non_blocking=True),
            seg_mask=self.seg_mask.to(device, non_blocking=True),
            prop_target={k: v.to(device, non_blocking=True)
                         for k, v in self.prop_target.items()},
            prop_mask=self.prop_mask.to(device, non_blocking=True),
            case_id=self.case_id,
        )


# ---------------------------------------------------------------------------
def _encode_clinical_len(clinical_vars: Sequence[str]) -> int:
    """Vector length produced by _encode_clinical for a given var list."""
    n = 0
    for name in clinical_vars:
        if name == "sex":
            n += 2
        elif name == "smoking_status":
            n += 3
        else:
            n += 1
    return n


def _encode_clinical(record: CaseRecord, clinical_vars: Sequence[str]) -> np.ndarray:
    """Simple dense encoding of clinical variables.

    - numeric fields are kept as-is, NaN → 0 with an unused indicator bit;
    - ``sex`` one-hot {M,F}; ``smoking_status`` one-hot {never,former,current}.
    The full vector length is derived from ``clinical_vars``.
    """
    vec: List[float] = []
    for name in clinical_vars:
        v = record.clinical.get(name, None)
        if name == "sex":
            vec.extend([float(v == "M"), float(v == "F")])
        elif name == "smoking_status":
            vec.extend([float(v == "never"),
                        float(v == "former"),
                        float(v == "current")])
        else:
            vec.append(0.0 if v is None or (isinstance(v, float) and np.isnan(v))
                       else float(v))
    return np.asarray(vec, dtype=np.float32)


def _rasterise_lesions(
    lesions: Sequence[LesionRecord],
    target_hw: int,
    dataset_root: Optional[str] = None,
) -> np.ndarray:
    """Build a 2-D lesion map from the NIfTI mask files.

    Loads each mask, takes a maximum-intensity projection across the axial
    (Z) axis, then resizes to ``(target_hw, target_hw)``.  The union of all
    lesion projections is returned as a binary float32 array.

    Falls back to an all-zeros map when ``dataset_root`` is None or no mask
    file exists on disk.
    """
    import nibabel as nib

    m = np.zeros((target_hw, target_hw), dtype=np.float32)
    if not lesions or dataset_root is None:
        return m
    for lez in lesions:
        mask_path = Path(dataset_root) / lez.mask_path
        if not mask_path.exists():
            continue
        vol = nib.load(str(mask_path)).get_fdata().astype(np.float32)
        proj = vol.max(axis=2)                          # max-project → (Y, X)
        proj_resized = _sk_resize(
            proj, (target_hw, target_hw),
            order=0, anti_aliasing=False, preserve_range=True,
        )
        m = np.maximum(m, (proj_resized > 0).astype(np.float32))
    return m


def _prop_targets(
    lesions: Sequence[LesionRecord],
) -> Dict[str, int]:
    """Case-level property labels = majority vote over annotated lesions."""
    out: Dict[str, int] = {}
    for group, classes in PROPERTY_GROUPS.items():
        votes: Dict[str, int] = {c: 0 for c in classes}
        for lez in lesions:
            v = getattr(lez.properties, group, None)
            if v in votes:
                votes[v] += 1
        winner = max(votes, key=lambda c: votes[c]) if lesions else classes[0]
        out[group] = classes.index(winner)
    return out


# ---------------------------------------------------------------------------
class BMDataset(Dataset):
    """Bach-Mai chest CT dataset with masked multitask supervision."""

    def __init__(
        self,
        cases: Sequence[CaseRecord],
        *,
        dataset_root: str,
        preproc: PreprocConfig,
        clinical_vars: Sequence[str],
        lesion_table_path: Optional[str] = None,
        n_keys: int = 16,
        transforms=None,
        disable_clinical: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.cases = list(cases)
        self.root = Path(dataset_root)
        self.preproc = preproc
        self.clinical_vars = list(clinical_vars)
        self.disable_clinical = bool(disable_clinical)
        self.n_keys = n_keys
        self.transforms = transforms
        self.cache_dir = cache_dir
        self.lesion_table: Dict[str, List[LesionRecord]] = (
            load_lesion_table(lesion_table_path) if lesion_table_path else {}
        )

    def __len__(self) -> int:
        return len(self.cases)

    def has_strong(self, rec: CaseRecord) -> bool:
        return rec.case_id in self.lesion_table

    def _case_lesions(self, rec: CaseRecord) -> List[LesionRecord]:
        return self.lesion_table.get(rec.case_id, [])

    def __getitem__(self, idx: int) -> dict:
        rec = self.cases[idx]
        nifti_path = str(self.root / rec.series_path)

        # -- image ----------------------------------------------------------
        #   ``rec.series_path`` points at a NIfTI file under ``images/``.
        #   Strong-annotation membership is derived from the lesion table; if
        #   any lesion mask for the case exists on disk we hand the *first*
        #   such mask to preprocess_case so the key-slice sampler steers
        #   toward lesion-bearing slices.
        lesions = self._case_lesions(rec)
        has_strong = bool(lesions)
        lesion_mask_path: Optional[str] = None
        for lez in lesions:
            candidate = self.root / lez.mask_path
            if candidate.exists():
                lesion_mask_path = str(candidate)
                break
        stack = preprocess_case(
            nifti_path,
            self.preproc,
            n_keys=self.n_keys,
            lesion_mask_path=lesion_mask_path,
            cache_dir=self.cache_dir,
        )                                                          # (K, C, Y, X)
        if self.transforms is not None:
            stack = self.transforms(stack)

        # -- strong supervision ---------------------------------------------
        seg_target = _rasterise_lesions(lesions, self.preproc.axial_crop,
                                         dataset_root=str(self.root))
        props = _prop_targets(lesions)

        if self.disable_clinical:
            clin_vec = np.zeros(_encode_clinical_len(self.clinical_vars), dtype=np.float32)
        else:
            clin_vec = _encode_clinical(rec, self.clinical_vars)
        return {
            "image":        torch.from_numpy(stack).float(),
            "case_label":   torch.tensor(rec.label, dtype=torch.long),
            "clinical":     torch.from_numpy(clin_vec),
            "seg_target":   torch.from_numpy(seg_target).unsqueeze(0),
            "seg_mask":     torch.tensor(float(has_strong)),
            "prop_target":  {k: torch.tensor(v, dtype=torch.long)
                             for k, v in props.items()},
            "prop_mask":    torch.tensor(float(has_strong)),
            "case_id":      rec.case_id,
        }


# ---------------------------------------------------------------------------
def multitask_collate(samples: Sequence[dict]) -> MultitaskBatch:
    """Collate fn that produces :class:`MultitaskBatch`."""
    img = torch.stack([s["image"] for s in samples])
    case_label = torch.stack([s["case_label"] for s in samples])
    clinical = torch.stack([s["clinical"] for s in samples])
    seg_target = torch.stack([s["seg_target"] for s in samples])
    seg_mask = torch.stack([s["seg_mask"] for s in samples])
    prop_mask = torch.stack([s["prop_mask"] for s in samples])
    prop_target = {
        g: torch.stack([s["prop_target"][g] for s in samples])
        for g in PROPERTY_GROUPS
    }
    return MultitaskBatch(
        image=img,
        case_label=case_label,
        clinical=clinical,
        seg_target=seg_target,
        seg_mask=seg_mask,
        prop_target=prop_target,
        prop_mask=prop_mask,
        case_id=[s["case_id"] for s in samples],
    )
