"""Paired image+mask augmentations for slice-level training.

The base transforms in ``transforms.py`` only operate on a single tensor.
Segmentation needs synchronised geometric transforms on both image and mask
(flip, rotate, shift), while intensity perturbations only touch the image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from scipy.ndimage import rotate as _rotate, shift as _shift

# (image, mask) -> (image, mask). image: (C,H,W) float32 in [0,1].
# mask: (1,H,W) float32 binary.
PairedTransform = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class PairedRandomFlip:
    p: float = 0.5
    horizontal: bool = True
    vertical: bool = False

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        rng = np.random.default_rng()
        if self.horizontal and rng.random() < self.p:
            img = np.ascontiguousarray(img[..., ::-1])
            mask = np.ascontiguousarray(mask[..., ::-1])
        if self.vertical and rng.random() < self.p:
            img = np.ascontiguousarray(img[..., ::-1, :])
            mask = np.ascontiguousarray(mask[..., ::-1, :])
        return img, mask


@dataclass
class PairedRandomRotate:
    max_deg: float = 15.0
    p: float = 0.5

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        rng = np.random.default_rng()
        if rng.random() >= self.p:
            return img, mask
        angle = float(rng.uniform(-self.max_deg, self.max_deg))
        img = _rotate(img, angle, axes=(-1, -2), reshape=False,
                       order=1, mode="constant", cval=0.0)
        mask = _rotate(mask, angle, axes=(-1, -2), reshape=False,
                        order=0, mode="constant", cval=0.0)
        return img.astype(np.float32), mask.astype(np.float32)


@dataclass
class PairedRandomShift:
    max_px: int = 16
    p: float = 0.5

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        rng = np.random.default_rng()
        if rng.random() >= self.p:
            return img, mask
        dy = int(rng.integers(-self.max_px, self.max_px + 1))
        dx = int(rng.integers(-self.max_px, self.max_px + 1))
        # use scipy shift with order=1 for image, order=0 for mask
        img = _shift(img, shift=(0, dy, dx), order=1,
                      mode="constant", cval=0.0)
        mask = _shift(mask, shift=(0, dy, dx), order=0,
                       mode="constant", cval=0.0)
        return img.astype(np.float32), mask.astype(np.float32)


@dataclass
class IntensityJitter:
    """Image-only intensity perturbation."""
    brightness: float = 0.05
    contrast: float = 0.05
    p: float = 0.5

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        rng = np.random.default_rng()
        if rng.random() >= self.p:
            return img, mask
        b = float(rng.uniform(-self.brightness, self.brightness))
        c = float(rng.uniform(1 - self.contrast, 1 + self.contrast))
        img = np.clip((img - 0.5) * c + 0.5 + b, 0.0, 1.0).astype(np.float32)
        return img, mask


class PairedCompose:
    def __init__(self, transforms: List[PairedTransform]):
        self.transforms = list(transforms)

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


def build_train_paired(rotate_deg: float = 15.0,
                       shift_px: int = 16,
                       brightness: float = 0.05,
                       contrast: float = 0.05) -> PairedCompose:
    return PairedCompose([
        PairedRandomFlip(p=0.5, horizontal=True, vertical=False),
        PairedRandomRotate(max_deg=rotate_deg, p=0.5),
        PairedRandomShift(max_px=shift_px, p=0.5),
        IntensityJitter(brightness=brightness, contrast=contrast, p=0.5),
    ])
