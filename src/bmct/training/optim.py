"""Optimizer and scheduler builders.

The train-default YAML structures these under ``optimizer:`` and
``scheduler:`` sub-blocks::

    optimizer:
      name: adamw
      lr: 1.0e-4
      weight_decay: 1.0e-4
    scheduler:
      name: cosine
      warmup_epochs: 5
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from omegaconf import DictConfig
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(params: Iterable[torch.nn.Parameter],
                    cfg: DictConfig) -> Optimizer:
    name = str(cfg.get("name", "adamw")).lower()
    if name == "adamw":
        return AdamW(
            params,
            lr=float(cfg.lr),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
        )
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer: Optimizer,
                    cfg: DictConfig,
                    total_epochs: int,
                    steps_per_epoch: int = 1):
    name = str(cfg.get("name", "cosine")).lower()
    warmup = int(cfg.get("warmup_epochs", 0))
    if name == "cosine":
        def fn(step: int) -> float:
            epoch = step / max(steps_per_epoch, 1)
            if epoch < warmup:
                return epoch / max(1, warmup)
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return LambdaLR(optimizer, fn)
    if name == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"Unknown schedule '{name}'")
