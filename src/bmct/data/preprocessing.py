"""NIfTI → tensor preprocessing pipeline.

The codebase is NIfTI-native: every case is stored as a single ``.nii`` or
``.nii.gz`` file and loaded with nibabel.  No DICOM tooling is required,
which makes the pipeline portable, test-friendly, and easy to share.

Pipeline stages:
    load_ct_volume     nibabel NIfTI loader → (volume, spacing_zyx)
    resample_isotropic trilinear resample to (slice_spacing, in_plane, in_plane)
    apply_windowing    HU clipping & min-max rescale to [0, 1] per window
    build_25d_stack    assemble N=stack_size adjacent slices per key slice
    crop_or_pad_axial  pad or centre-crop the axial plane to a fixed size

All numeric parameters come from ``configs/data/*.yaml`` so the same code
reproduces every cohort, including the toy dataset under
``dataset/toy/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib                 # pragma: no cover
except ImportError:                       # allow import without nibabel for docs
    nib = None

try:
    from scipy.ndimage import zoom        # pragma: no cover
except ImportError:
    zoom = None


# ---------------------------------------------------------------------------
@dataclass
class PreprocConfig:
    in_plane_spacing_mm: float = 0.7
    slice_spacing_mm: float = 1.5
    lung_window: Tuple[float, float] = (-1000.0, -400.0)
    mediastinal_window: Tuple[float, float] = (-150.0, 250.0)
    # When set, replaces the dual lung/mediastinal channels with a single
    # visible-region window rescaled to [0,1] → 1 channel.
    visible_window: Optional[Tuple[float, float]] = None
    stack_size: int = 3
    axial_crop: int = 352


# ---------------------------------------------------------------------------
def load_ct_volume(nifti_path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a NIfTI CT volume.

    Returns ``(volume, spacing_zyx)`` where ``volume`` is a ``(Z, Y, X)`` float32
    array in Hounsfield units (after applying the NIfTI scale slope/intercept
    if present) and ``spacing_zyx`` is the voxel size in mm.

    Accepts ``.nii``, ``.nii.gz`` — anything ``nibabel.load`` understands.
    """
    if nib is None:
        raise ImportError("nibabel is required for load_ct_volume(); "
                           "pip install nibabel")
    img = nib.load(str(nifti_path))
    # get_fdata applies scl_slope / scl_inter from the header automatically.
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    # nibabel gives (X, Y, Z) in voxel order for canonical NIfTI; transpose to (Z, Y, X).
    volume = np.transpose(arr, (2, 1, 0))
    # Auto-HU offset detection. A proper chest CT volume has many negative
    # voxels (air outside body + lung parenchyma ~-700 to -900 HU): typically
    # 60--85% of voxels are negative. Two failure modes we have observed in
    # this cohort:
    #   (1) `vmin >= 0`: scanner / export stripped the -1024 intercept, so
    #       everything is in [0, ~4000].
    #   (2) `vmin == -2000` (or -1024) but the body tissue is still shifted
    #       up by ~1024: a background fill at the legacy minimum coexists
    #       with shifted body values, so `vmin` alone misses the problem.
    # Using `frac_neg < 0.3` catches both — legitimate HU volumes (including
    # narrow-FOV non-thoracic scans) keep `frac_neg >= 0.6`, while shifted
    # volumes drop to <0.25.
    if float((volume < 0).mean()) < 0.30:
        volume = volume - 1024.0
    zooms = img.header.get_zooms()[:3]                      # (x, y, z) mm
    spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return volume, spacing_zyx


def load_mask(nifti_path: str) -> np.ndarray:
    """Load a binary NIfTI mask in (Z, Y, X) order."""
    if nib is None:
        raise ImportError("nibabel is required for load_mask().")
    img = nib.load(str(nifti_path))
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    return np.transpose(arr, (2, 1, 0)) > 0.5


# ---------------------------------------------------------------------------
def resample_isotropic(
    volume: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    in_plane_spacing_mm: float,
    slice_spacing_mm: float,
    order: int = 1,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Trilinear resample to the requested (z, y, x) spacing.

    ``order`` passes through to ``scipy.ndimage.zoom`` — use 1 for images,
    0 for binary masks.
    """
    if zoom is None:
        raise ImportError("scipy is required for resample_isotropic().")
    target = (slice_spacing_mm, in_plane_spacing_mm, in_plane_spacing_mm)
    factors = tuple(spacing_zyx[i] / target[i] for i in range(3))
    resampled = zoom(volume, factors, order=order, mode="nearest",
                      grid_mode=False, prefilter=(order > 1))
    return resampled.astype(volume.dtype), target


# ---------------------------------------------------------------------------
def apply_windowing(
    volume: np.ndarray,
    windows: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Apply one or more HU windows and return a (C, Z, Y, X) float32 array.

    Each window yields one channel rescaled to [0, 1].
    """
    volume = volume.astype(np.float32)
    channels = []
    for lo, hi in windows:
        clipped = np.clip(volume, lo, hi)
        channels.append((clipped - lo) / max(hi - lo, 1e-6))
    return np.stack(channels, axis=0)


# ---------------------------------------------------------------------------
def build_25d_stack(
    volume_chw_zyx: np.ndarray,
    key_indices: Sequence[int],
    stack_size: int = 3,
) -> np.ndarray:
    """Assemble a 2.5-D tensor of shape (K, C*stack_size, Y, X).

    ``volume_chw_zyx`` is a (C, Z, Y, X) array (e.g. output of
    :func:`apply_windowing`). For each index ``k`` in ``key_indices`` we take
    ``stack_size`` adjacent slices centred at ``k`` and concatenate them along
    the channel axis.
    """
    assert stack_size >= 1 and stack_size % 2 == 1, "stack_size must be odd"
    half = stack_size // 2
    C, Z, Y, X = volume_chw_zyx.shape
    out = np.empty((len(key_indices), C * stack_size, Y, X), dtype=np.float32)
    for i, k in enumerate(key_indices):
        lo, hi = k - half, k + half + 1
        slab = np.stack([
            volume_chw_zyx[:, max(0, min(Z - 1, z))] for z in range(lo, hi)
        ], axis=1)                                      # (C, stack, Y, X)
        out[i] = slab.reshape(C * stack_size, Y, X)
    return out


# ---------------------------------------------------------------------------
def crop_or_pad_axial(
    stack: np.ndarray,
    target: int = 352,
) -> np.ndarray:
    """Centre-crop or zero-pad the last two axes to ``target``."""
    *lead, Y, X = stack.shape
    out_shape = (*lead, target, target)
    out = np.zeros(out_shape, dtype=stack.dtype)

    def _align(src_len: int, dst_len: int) -> Tuple[int, int, int, int]:
        if src_len >= dst_len:                       # crop
            start = (src_len - dst_len) // 2
            return start, start + dst_len, 0, dst_len
        pad = (dst_len - src_len) // 2
        return 0, src_len, pad, pad + src_len

    sy0, sy1, dy0, dy1 = _align(Y, target)
    sx0, sx1, dx0, dx1 = _align(X, target)
    out[..., dy0:dy1, dx0:dx1] = stack[..., sy0:sy1, sx0:sx1]
    return out


# ---------------------------------------------------------------------------
def select_key_slices(
    n_slices: int,
    lesion_mask: np.ndarray | None = None,
    n_keys: int = 16,
) -> List[int]:
    """Choose ``n_keys`` slice indices.

    If a binary lesion mask is supplied we return the lesion-bearing slices
    (plus uniform padding up to ``n_keys``); otherwise we sample uniformly
    across the scan.  Case-level training uses the uniform branch.
    """
    if lesion_mask is not None:
        nz = np.where(lesion_mask.any(axis=(1, 2)))[0].tolist()
        if len(nz) >= n_keys:
            return [int(x) for x in np.linspace(nz[0], nz[-1], n_keys).round()]
        pad = np.linspace(0, n_slices - 1, n_keys - len(nz)).round().astype(int)
        combined = sorted(set(int(x) for x in (*nz, *pad)))
        if len(combined) < n_keys:
            # Volume too short for dedup to yield n_keys unique indices; fall back.
            return [int(x) for x in np.linspace(0, n_slices - 1, n_keys).round()]
        return combined
    return [int(x) for x in np.linspace(0, n_slices - 1, n_keys).round()]


# ---------------------------------------------------------------------------
def _cache_key(cfg: PreprocConfig, n_keys: int, with_mask: bool) -> str:
    """Stable key encoding all preproc parameters that affect the output stack."""
    if cfg.visible_window is not None:
        win = (f"vw{cfg.visible_window[0]:.0f}_{cfg.visible_window[1]:.0f}")
    else:
        win = (f"lw{cfg.lung_window[0]:.0f}_{cfg.lung_window[1]:.0f}_"
               f"mw{cfg.mediastinal_window[0]:.0f}_{cfg.mediastinal_window[1]:.0f}")
    return (f"k{n_keys}_s{cfg.stack_size}_c{cfg.axial_crop}_"
            f"sp{cfg.in_plane_spacing_mm}x{cfg.slice_spacing_mm}_"
            f"{win}_m{int(with_mask)}")


def preprocess_case(
    nifti_path: str,
    cfg: PreprocConfig,
    n_keys: int = 16,
    lesion_mask_path: str | None = None,
    cache_dir: str | None = None,
) -> np.ndarray:
    """End-to-end: NIfTI file → (K, C*stack, target, target) float32.

    ``nifti_path`` is a single ``.nii`` or ``.nii.gz`` file.  ``lesion_mask_path``
    is optional; when supplied it steers the key-slice sampler toward lesion-
    bearing slices (strong supervision cases only).

    When ``cache_dir`` is supplied, the post-preprocess stack is stored as
    ``float16`` at ``<cache_dir>/<key>/<case_id>.npy`` and reused on later
    calls.  The cache key encodes every parameter that affects the output, so
    changing spacing / window / n_keys forces a fresh cache silo.
    """
    if cache_dir is not None:
        from pathlib import Path as _P
        key = _cache_key(cfg, n_keys, with_mask=lesion_mask_path is not None)
        case_id = _P(nifti_path).name.replace(".nii.gz", "").replace(".nii", "")
        cache_path = _P(cache_dir) / key / f"{case_id}.npy"
        if cache_path.exists():
            return np.load(cache_path).astype(np.float32)

    volume, spacing = load_ct_volume(nifti_path)
    volume, _ = resample_isotropic(
        volume, spacing,
        cfg.in_plane_spacing_mm, cfg.slice_spacing_mm,
        order=1,
    )
    if cfg.visible_window is not None:
        channels = apply_windowing(volume, [cfg.visible_window])
    else:
        channels = apply_windowing(volume, [cfg.lung_window, cfg.mediastinal_window])
    lesion_mask = None
    if lesion_mask_path is not None and Path(lesion_mask_path).exists():
        raw = load_mask(lesion_mask_path).astype(np.float32)
        # match the resampled image geometry
        raw_res, _ = resample_isotropic(
            raw, spacing,
            cfg.in_plane_spacing_mm, cfg.slice_spacing_mm,
            order=0,
        )
        lesion_mask = (raw_res > 0.5).astype(np.uint8)
    key_idx = select_key_slices(volume.shape[0], lesion_mask, n_keys)
    stack = build_25d_stack(channels, key_idx, cfg.stack_size)
    stack = crop_or_pad_axial(stack, cfg.axial_crop)

    if cache_dir is not None:
        from pathlib import Path as _P
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write — pass a file object so np.save doesn't append .npy.
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, stack.astype(np.float16))
        tmp.rename(cache_path)
    return stack
