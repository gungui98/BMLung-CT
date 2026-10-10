"""Lightweight numpy-based augmentations used during training.

The dataset returns (K, C, Y, X) float32 tensors in [0, 1]; augmentations are
applied in numpy before the final ``torch.from_numpy`` so that they compose
nicely with multi-worker DataLoading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


Transform = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
@dataclass
class RandomFlip:
    p: float = 0.5
    axis: int = -1                   # horizontal flip on x by default
    rng: np.random.Generator | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rng = self.rng or np.random.default_rng()
        if rng.random() < self.p:
            x = np.flip(x, axis=self.axis).copy()
        return x


@dataclass
class RandomIntensityJitter:
    brightness: float = 0.05
    contrast: float = 0.05
    rng: np.random.Generator | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rng = self.rng or np.random.default_rng()
        b = rng.uniform(-self.brightness, self.brightness)
        c = rng.uniform(1 - self.contrast, 1 + self.contrast)
        return np.clip((x - 0.5) * c + 0.5 + b, 0.0, 1.0).astype(np.float32)


@dataclass
class RandomAxialShift:
    max_shift_px: int = 16
    rng: np.random.Generator | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rng = self.rng or np.random.default_rng()
        dy = int(rng.integers(-self.max_shift_px, self.max_shift_px + 1))
        dx = int(rng.integers(-self.max_shift_px, self.max_shift_px + 1))
        return np.roll(x, shift=(dy, dx), axis=(-2, -1))


# ---------------------------------------------------------------------------
class Compose:
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


# ---------------------------------------------------------------------------
def build_train_transforms(seed: int = 2026) -> Compose:
    rng = np.random.default_rng(seed)
    return Compose([
        RandomFlip(p=0.5, axis=-1, rng=rng),
        RandomIntensityJitter(brightness=0.04, contrast=0.05, rng=rng),
        RandomAxialShift(max_shift_px=12, rng=rng),
    ])


def build_eval_transforms() -> Compose:
    return Compose([])
