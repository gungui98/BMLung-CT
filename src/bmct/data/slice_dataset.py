"""Slice-level dataset for the kc-cancer-final cohort.

The 2.5-D / MIL pipeline (see ``BMDataset``) treats each *case* as one bag of
key-slices. Because kc-cancer-final ships ROI-cropped sub-volumes (Z ≈ 10–17
slices, all annotated), MIL attention has nothing to focus on. This dataset
collapses to the slice level: every axial slice of every case is a separate
training sample. Cases without lesion annotations contribute negative slices.

Pipeline per case (run once, cached to ``<cache_dir>/<key>/<case_id>.npz``):
    load_ct_volume          NIfTI → (Z, Y, X)  HU
    resample_isotropic      → target voxel size
    apply_windowing         → (1, Z, Y, X) ∈ [0, 1]  (visible window only)
    crop_or_pad_axial       → (Z, axial_crop, axial_crop)
    mask: union of all lesion NIfTIs.
    attr: (Z, K_total) multi-hot vector. For each slice z, OR over the
          one-hot attribute encoding of every lesion whose mask intersects z.

__getitem__ memory-maps the npz and returns one slice + its attr vector.
``attr_mask`` is 1 iff the slice itself contains any lesion pixel — i.e. the
attribute loss should only be applied on lesion-bearing slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from bmct.data.preprocessing import (
    PreprocConfig,
    apply_windowing,
    crop_or_pad_axial,
    load_ct_volume,
    load_mask,
    resample_isotropic,
)
from bmct.data.schema import PROPERTY_GROUPS, CaseRecord, LesionRecord


# ---------------------------------------------------------------------------
# Attribute encoding helpers
# ---------------------------------------------------------------------------
def build_group_offsets() -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Layout of property classes in the flat K_total multi-hot vector.

    Returns:
        offsets: group → (start, end) indices in the K_total vector.
        K_total: sum of class counts across all groups.
    """
    off: Dict[str, Tuple[int, int]] = {}
    cur = 0
    for g, classes in PROPERTY_GROUPS.items():
        off[g] = (cur, cur + len(classes))
        cur += len(classes)
    return off, cur


GROUP_OFFSETS, K_TOTAL = build_group_offsets()


def lesion_onehot(lez: LesionRecord) -> np.ndarray:
    """Return the K_total-length one-hot vector for one lesion's attributes.

    Missing / out-of-vocab values are silently skipped (their class bit stays
    0); the OR-aggregation across lesions in the slice still produces a valid
    multi-hot target.
    """
    v = np.zeros(K_TOTAL, dtype=np.float32)
    for g, classes in PROPERTY_GROUPS.items():
        val = getattr(lez.properties, g, None)
        if val in classes:
            start, _ = GROUP_OFFSETS[g]
            v[start + classes.index(val)] = 1.0
    return v


# ---------------------------------------------------------------------------
def _cache_key(cfg: PreprocConfig) -> str:
    win = cfg.visible_window or cfg.lung_window
    return (f"slice_v2_sp{cfg.in_plane_spacing_mm}x{cfg.slice_spacing_mm}"
            f"_c{cfg.axial_crop}_vw{win[0]:.0f}_{win[1]:.0f}")


def _preprocess_case_to_disk(
    case: CaseRecord,
    lesions: Sequence[LesionRecord],
    dataset_root: Path,
    cfg: PreprocConfig,
    cache_path: Path,
) -> None:
    """Run the per-case preprocessing once and dump img / mask / attr arrays.

    attr is shape ``(Z, K_total)``: for slice z, OR over the one-hot vector of
    each lesion whose mask intersects that slice.
    """
    volume, spacing = load_ct_volume(str(dataset_root / case.series_path))
    volume_r, _ = resample_isotropic(
        volume, spacing,
        cfg.in_plane_spacing_mm, cfg.slice_spacing_mm, order=1,
    )
    win = cfg.visible_window or cfg.lung_window
    channels = apply_windowing(volume_r, [win])           # (1, Z, Y, X) float32
    img_stack = channels[0]                                # (Z, Y, X)
    img_stack = crop_or_pad_axial(img_stack, cfg.axial_crop)

    Z, Y, X = volume_r.shape
    union = np.zeros((Z, Y, X), dtype=np.float32)
    attr_stack = np.zeros((Z, K_TOTAL), dtype=np.float32)

    for lez in lesions:
        mpath = dataset_root / lez.mask_path
        if not mpath.exists():
            continue
        raw = load_mask(str(mpath)).astype(np.float32)
        raw_r, _ = resample_isotropic(
            raw, spacing,
            cfg.in_plane_spacing_mm, cfg.slice_spacing_mm, order=0,
        )
        rZ, rY, rX = raw_r.shape
        z = min(rZ, Z); y = min(rY, Y); x = min(rX, X)
        sub = raw_r[:z, :y, :x]
        # union of lesion masks (for segmentation target)
        union[:z, :y, :x] = np.maximum(union[:z, :y, :x], sub)
        # per-slice multi-hot OR for attributes
        onehot = lesion_onehot(lez)
        has_lesion_in_slice = sub.reshape(z, -1).any(axis=1)  # (z,) bool
        for zi in np.flatnonzero(has_lesion_in_slice):
            attr_stack[zi] = np.maximum(attr_stack[zi], onehot)

    mask_stack = (union > 0.5).astype(np.uint8)
    mask_stack = crop_or_pad_axial(mask_stack, cfg.axial_crop)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez(str(tmp),
             img=img_stack.astype(np.float16),
             mask=mask_stack,
             attr=attr_stack)
    tmp.rename(cache_path)


# ---------------------------------------------------------------------------
@dataclass
class SliceBatch:
    image: torch.Tensor          # (B, 1, Y, X)
    seg_target: torch.Tensor     # (B, 1, Y, X)
    case_label: torch.Tensor     # (B,)
    seg_mask: torch.Tensor       # (B,) 1 if case has any lesion annotation
    prop_target: torch.Tensor    # (B, K_total) multi-hot attr labels
    attr_mask: torch.Tensor      # (B,) 1 if THIS slice has any lesion pixel
    case_id: List[str]
    slice_idx: List[int]

    def to(self, device) -> "SliceBatch":
        return SliceBatch(
            image=self.image.to(device, non_blocking=True),
            seg_target=self.seg_target.to(device, non_blocking=True),
            case_label=self.case_label.to(device, non_blocking=True),
            seg_mask=self.seg_mask.to(device, non_blocking=True),
            prop_target=self.prop_target.to(device, non_blocking=True),
            attr_mask=self.attr_mask.to(device, non_blocking=True),
            case_id=self.case_id,
            slice_idx=self.slice_idx,
        )


# ---------------------------------------------------------------------------
class BMSliceDataset(Dataset):
    """One sample = one axial slice from one case.

    Returns per-slice img + seg mask + multi-hot attr vector. Slices from
    cases without lesion annotations contribute all-zero seg + attr (true
    negatives).
    """

    def __init__(
        self,
        cases: Sequence[CaseRecord],
        *,
        dataset_root: str,
        preproc: PreprocConfig,
        lesion_table: Dict[str, List[LesionRecord]],
        cache_dir: str,
        precompute: bool = True,
        transforms=None,
        stack_2_5d: int = 1,
    ) -> None:
        """``stack_2_5d``: number of adjacent slices stacked into the channel
        dim (must be odd). 1 = 1-channel (current default). 3 = use slice
        z-1, z, z+1 → 3-channel input."""
        assert stack_2_5d >= 1 and stack_2_5d % 2 == 1, "stack_2_5d must be odd"
        self.cases = list(cases)
        self.root = Path(dataset_root)
        self.preproc = preproc
        self.lesion_table = lesion_table
        self.cache_root = Path(cache_dir) / _cache_key(preproc)
        self.transforms = transforms
        self.stack_2_5d = stack_2_5d

        import nibabel as nib

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._index: List[Tuple[int, int]] = []

        for i, c in enumerate(self.cases):
            cache_path = self.cache_root / f"{c.case_id}.npz"
            if cache_path.exists():
                with np.load(cache_path) as npz:
                    z = int(npz["img"].shape[0])
            else:
                img = nib.load(str(self.root / c.series_path))
                Z_raw = int(img.shape[2])
                z_raw = float(img.header.get_zooms()[2])
                z = max(1, int(round(Z_raw * z_raw / preproc.slice_spacing_mm)))
            for zi in range(z):
                self._index.append((i, zi))

        if precompute:
            self._precompute_all()

        self._mmap: Dict[str, np.lib.npyio.NpzFile] = {}

    # ------------------------------------------------------------------
    def _precompute_all(self, n_workers: int = 8) -> None:
        from tqdm.auto import tqdm
        todo = [c for c in self.cases
                if not (self.cache_root / f"{c.case_id}.npz").exists()]
        if not todo:
            return
        if n_workers <= 1:
            for c in tqdm(todo, desc=f"preprocess→{self.cache_root.name}"):
                lesions = self.lesion_table.get(c.case_id, [])
                _preprocess_case_to_disk(
                    c, lesions, self.root, self.preproc,
                    self.cache_root / f"{c.case_id}.npz",
                )
            return

        # Parallel: each worker preprocesses one case at a time.
        import concurrent.futures as _cf
        jobs = [(c, self.lesion_table.get(c.case_id, []),
                 self.root, self.preproc,
                 self.cache_root / f"{c.case_id}.npz")
                for c in todo]
        with _cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_preprocess_case_to_disk, *j) for j in jobs]
            for _ in tqdm(_cf.as_completed(futures), total=len(futures),
                          desc=f"preprocess→{self.cache_root.name}"):
                pass

    def _open(self, case_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if case_id not in self._mmap:
            self._mmap[case_id] = np.load(
                self.cache_root / f"{case_id}.npz", mmap_mode="r",
            )
        npz = self._mmap[case_id]
        # Some npz from older versions may miss "attr"; fall back to zeros.
        attr = npz["attr"] if "attr" in npz.files else np.zeros(
            (npz["img"].shape[0], K_TOTAL), dtype=np.float32)
        return npz["img"], npz["mask"], attr

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ci, zi = self._index[idx]
        case = self.cases[ci]
        img_stack, mask_stack, attr_stack = self._open(case.case_id)
        Z = img_stack.shape[0]
        zi = min(zi, Z - 1)
        if self.stack_2_5d > 1:
            half = self.stack_2_5d // 2
            zs = [max(0, min(Z - 1, zi + d)) for d in range(-half, half + 1)]
            img = np.stack([np.asarray(img_stack[k], dtype=np.float32)
                            for k in zs], axis=0)         # (stack, Y, X)
        else:
            img = np.asarray(img_stack[zi], dtype=np.float32)[None, ...]
        msk = np.asarray(mask_stack[zi], dtype=np.float32)[None, ...]
        attr = np.asarray(attr_stack[zi], dtype=np.float32)
        if self.transforms is not None:
            img, msk = self.transforms(img, msk)
        attr_mask = float(msk.any())
        return {
            "image":      torch.from_numpy(img),
            "seg_target": torch.from_numpy(msk),
            "prop_target": torch.from_numpy(attr),
            "attr_mask":  torch.tensor(attr_mask, dtype=torch.float32),
            "case_label": torch.tensor(int(case.label), dtype=torch.long),
            "seg_mask":   torch.tensor(
                float(case.case_id in self.lesion_table), dtype=torch.float32,
            ),
            "case_id":   case.case_id,
            "slice_idx": int(zi),
        }


def slice_collate(samples: Sequence[dict]) -> SliceBatch:
    return SliceBatch(
        image=torch.stack([s["image"] for s in samples]),
        seg_target=torch.stack([s["seg_target"] for s in samples]),
        case_label=torch.stack([s["case_label"] for s in samples]),
        seg_mask=torch.stack([s["seg_mask"] for s in samples]),
        prop_target=torch.stack([s["prop_target"] for s in samples]),
        attr_mask=torch.stack([s["attr_mask"] for s in samples]),
        case_id=[s["case_id"] for s in samples],
        slice_idx=[s["slice_idx"] for s in samples],
    )
