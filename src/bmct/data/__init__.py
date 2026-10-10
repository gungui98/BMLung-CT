"""Data subpackage: manifests, preprocessing, datasets, samplers.

All CT volumes are assumed to live on disk as ``.nii`` / ``.nii.gz`` files
and are loaded with nibabel — there is no DICOM path in this codebase.
"""
from bmct.data.schema import CaseRecord, LesionRecord, PropertyLabels
from bmct.data.splits import (
    build_splits, build_temporal_splits, build_external_partitions,
)
from bmct.data.preprocessing import (
    PreprocConfig,
    load_ct_volume,
    load_mask,
    resample_isotropic,
    apply_windowing,
    build_25d_stack,
    crop_or_pad_axial,
    preprocess_case,
)
from bmct.data.dataset import BMDataset, MultitaskBatch
from bmct.data.slice_dataset import BMSliceDataset, SliceBatch, slice_collate
from bmct.data.samplers import MaskedMultitaskSampler
from bmct.data.transforms import build_train_transforms, build_eval_transforms

__all__ = [
    "CaseRecord",
    "LesionRecord",
    "PropertyLabels",
    "build_splits",
    "build_temporal_splits",
    "build_external_partitions",
    "PreprocConfig",
    "load_ct_volume",
    "load_mask",
    "resample_isotropic",
    "apply_windowing",
    "build_25d_stack",
    "crop_or_pad_axial",
    "preprocess_case",
    "BMDataset",
    "MultitaskBatch",
    "BMSliceDataset",
    "SliceBatch",
    "slice_collate",
    "MaskedMultitaskSampler",
    "build_train_transforms",
    "build_eval_transforms",
]
