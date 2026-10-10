"""Per-nodule crop dataset for stage-2 attribute / cancer classification.

One sample = one annotated lesion. For training we crop a fixed-size patch
(centred on the lesion's 3-D centroid) from the cached preprocessed volume
of its case, with a few adjacent slices stacked into the channel dim.

Targets per sample:
    density      ∈ {solid, part_solid, ground_glass}                  (3-way; "consolidation" dropped — 0 train+val instances)
    margin       ∈ {smooth, lobulated, spiculated, irregular}         (4-way)
    cavitation   ∈ {absent, present}                                  (binary)
    cancer       ∈ {0, 1}                                             (binary, from case.label)

``lobe`` and ``inflammatory_pattern`` are intentionally dropped — kc-cancer-final
is ROI-cropped, so neither global lobe localisation nor large-FOV inflammation
patterns are learnable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from bmct.data.preprocessing import (
    PreprocConfig, apply_windowing, crop_or_pad_axial,
    load_ct_volume, load_mask, resample_isotropic,
)
from bmct.data.schema import CaseRecord, LesionRecord


# Classes (mutually exclusive per lesion, but stored as integer indices).
DENSITY_CLASSES   = ["solid", "part_solid", "ground_glass"]
MARGIN_CLASSES    = ["smooth", "lobulated", "spiculated", "irregular"]
CAVIT_CLASSES     = ["absent", "present"]


def _cache_key(cfg: PreprocConfig, crop_px: int) -> str:
    win = cfg.visible_window or cfg.lung_window
    return (f"nodule_v1_sp{cfg.in_plane_spacing_mm}x{cfg.slice_spacing_mm}"
            f"_crop{crop_px}_vw{win[0]:.0f}_{win[1]:.0f}")


def _build_one_nodule(case: CaseRecord, lez: LesionRecord,
                       dataset_root: Path, cfg: PreprocConfig,
                       crop_px: int) -> Optional[dict]:
    """Run preproc + crop for one lesion. Returns a dict with cached arrays."""
    volume, spacing = load_ct_volume(str(dataset_root / case.series_path))
    volume_r, _ = resample_isotropic(
        volume, spacing,
        cfg.in_plane_spacing_mm, cfg.slice_spacing_mm, order=1,
    )
    win = cfg.visible_window or cfg.lung_window
    channels = apply_windowing(volume_r, [win])[0]            # (Z,Y,X)

    mpath = dataset_root / lez.mask_path
    if not mpath.exists():
        return None
    raw = load_mask(str(mpath)).astype(np.float32)
    raw_r, _ = resample_isotropic(
        raw, spacing,
        cfg.in_plane_spacing_mm, cfg.slice_spacing_mm, order=0,
    )

    Z, Y, X = volume_r.shape
    rZ, rY, rX = raw_r.shape
    z = min(rZ, Z); y = min(rY, Y); x = min(rX, X)
    mask = np.zeros((Z, Y, X), dtype=np.float32)
    mask[:z, :y, :x] = raw_r[:z, :y, :x]

    if not mask.any() and raw.any():
        # Nearest-neighbour resampling can drop a lesion entirely when it lives
        # on one or two slices that fall between the target grid points: on
        # KC_CT_0052 the L4 lesion goes from 2,043 voxels to none when 12 slices
        # at 1.0 mm become 8 at 1.5 mm. Silently returning None here is what
        # removed such lesions from the benchmark cohort. Fall back to linear
        # resampling with an any-coverage threshold, which keeps the lesion on
        # the nearest slice. This only fires where the crop would otherwise not
        # exist, so no already-cached crop changes.
        raw_r, _ = resample_isotropic(
            raw, spacing,
            cfg.in_plane_spacing_mm, cfg.slice_spacing_mm, order=1,
        )
        rZ, rY, rX = raw_r.shape
        z = min(rZ, Z); y = min(rY, Y); x = min(rX, X)
        mask = np.zeros((Z, Y, X), dtype=np.float32)
        mask[:z, :y, :x] = (raw_r[:z, :y, :x] > 0).astype(np.float32)

    if not mask.any():
        return None

    # Find 3-D centroid + bbox of the lesion.
    nz = np.argwhere(mask > 0.5)
    cz, cy, cx = nz.mean(axis=0).astype(int)

    half = crop_px // 2
    # Pad image + mask so the crop is always valid.
    pad = ((1, 1), (half, half), (half, half))
    img_pad = np.pad(channels, pad, mode="constant")
    mask_pad = np.pad(mask, pad, mode="constant")
    zp = cz + 1; yp = cy + half; xp = cx + half

    # 3-channel stack: (z-1, z, z+1) centred slices.
    img_crop = img_pad[zp - 1: zp + 2,
                       yp - half: yp + half,
                       xp - half: xp + half]              # (3, crop, crop)
    mask_crop = mask_pad[zp,
                          yp - half: yp + half,
                          xp - half: xp + half]            # (crop, crop)

    return {
        "img":  img_crop.astype(np.float16),
        "mask": (mask_crop > 0.5).astype(np.uint8),
    }


def _safe_idx(name: str, classes: List[str]) -> int:
    """Class index, or -1 if name not in vocab (used as "ignore" marker)."""
    try:
        return classes.index(name) if name else -1
    except ValueError:
        return -1


@dataclass
class NoduleBatch:
    image: torch.Tensor          # (B, 3, H, W)
    mask:  torch.Tensor          # (B, 1, H, W)
    density: torch.Tensor        # (B,) long, -1 = ignore
    margin: torch.Tensor         # (B,) long, -1 = ignore
    cavit: torch.Tensor          # (B,) float 0/1, NaN = ignore
    cancer: torch.Tensor         # (B,) float 0/1 (case-level)
    case_id: List[str]
    lesion_id: List[str]

    def to(self, device) -> "NoduleBatch":
        return NoduleBatch(
            image=self.image.to(device, non_blocking=True),
            mask=self.mask.to(device, non_blocking=True),
            density=self.density.to(device, non_blocking=True),
            margin=self.margin.to(device, non_blocking=True),
            cavit=self.cavit.to(device, non_blocking=True),
            cancer=self.cancer.to(device, non_blocking=True),
            case_id=self.case_id,
            lesion_id=self.lesion_id,
        )


class NoduleDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[CaseRecord],
        lesion_table: Dict[str, List[LesionRecord]],
        *,
        dataset_root: str,
        preproc: PreprocConfig,
        cache_dir: str,
        crop_px: int = 64,
        transforms=None,
        precompute: bool = True,
    ) -> None:
        self.root = Path(dataset_root)
        self.preproc = preproc
        self.crop_px = crop_px
        self.transforms = transforms
        self.cache_root = Path(cache_dir) / _cache_key(preproc, crop_px)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Build flat lesion index — only include cases in this split that
        # have at least one lesion row in lesion_table.
        case_by_id = {c.case_id: c for c in cases}
        self.entries: List[Tuple[CaseRecord, LesionRecord, Path]] = []
        for cid, lesions in lesion_table.items():
            if cid not in case_by_id:
                continue
            case = case_by_id[cid]
            for lez in lesions:
                cache_path = self.cache_root / f"{cid}_{lez.lesion_id}.npz"
                self.entries.append((case, lez, cache_path))

        if precompute:
            self._precompute_all()

        # Filter entries to ones whose cache file actually exists
        # (some lesions have no valid mask on disk → no npz produced).
        self.entries = [e for e in self.entries if e[2].exists()]
        self._mmap: Dict[str, np.lib.npyio.NpzFile] = {}

    # ------------------------------------------------------------------
    def _precompute_all(self, n_workers: int = 8) -> None:
        from tqdm.auto import tqdm
        todo = [(c, l, p) for (c, l, p) in self.entries if not p.exists()]
        if not todo:
            return
        if n_workers <= 1:
            for c, l, p in tqdm(todo, desc=f"nodule→{self.cache_root.name}"):
                _save(c, l, p, self.root, self.preproc, self.crop_px)
            return
        import concurrent.futures as _cf
        with _cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_save, c, l, p, self.root,
                                    self.preproc, self.crop_px)
                       for c, l, p in todo]
            for _ in tqdm(_cf.as_completed(futures), total=len(futures),
                          desc=f"nodule→{self.cache_root.name}"):
                pass

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        case, lez, path = self.entries[idx]
        if str(path) not in self._mmap:
            self._mmap[str(path)] = np.load(path, mmap_mode="r")
        npz = self._mmap[str(path)]
        img = np.asarray(npz["img"], dtype=np.float32)          # (3, H, W)
        msk = np.asarray(npz["mask"], dtype=np.float32)[None]   # (1, H, W)
        if self.transforms is not None:
            img, msk = self.transforms(img, msk)
        return {
            "image":   torch.from_numpy(img),
            "mask":    torch.from_numpy(msk),
            "density": torch.tensor(
                _safe_idx(getattr(lez.properties, "density", None) or "",
                           DENSITY_CLASSES), dtype=torch.long,
            ),
            "margin":  torch.tensor(
                _safe_idx(getattr(lez.properties, "margin", None) or "",
                           MARGIN_CLASSES), dtype=torch.long,
            ),
            "cavit":   torch.tensor(
                float(getattr(lez.properties, "cavitation", "") == "present")
                if getattr(lez.properties, "cavitation", None)
                else float("nan"),
                dtype=torch.float32,
            ),
            "cancer":  torch.tensor(float(int(case.label)),
                                     dtype=torch.float32),
            "case_id":   case.case_id,
            "lesion_id": lez.lesion_id,
        }


def _save(case, lez, path, root, preproc, crop_px) -> None:
    out = _build_one_nodule(case, lez, root, preproc, crop_px)
    if out is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(str(tmp), **out)
    tmp.rename(path)


def nodule_collate(samples: Sequence[dict]) -> NoduleBatch:
    return NoduleBatch(
        image=torch.stack([s["image"] for s in samples]),
        mask=torch.stack([s["mask"] for s in samples]),
        density=torch.stack([s["density"] for s in samples]),
        margin=torch.stack([s["margin"] for s in samples]),
        cavit=torch.stack([s["cavit"] for s in samples]),
        cancer=torch.stack([s["cancer"] for s in samples]),
        case_id=[s["case_id"] for s in samples],
        lesion_id=[s["lesion_id"] for s in samples],
    )
