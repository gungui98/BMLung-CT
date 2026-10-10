"""Small JSON/YAML/filesystem helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_yaml(path: str):
    """Load an OmegaConf config, resolving ``defaults:`` composition if needed.

    For simple compositions we recursively merge every file referenced by
    ``defaults:`` before returning.  This mirrors Hydra semantics without
    requiring Hydra at runtime.
    """
    cfg = OmegaConf.load(path)
    base_dir = Path(path).parent
    if "defaults" in cfg:
        merged = OmegaConf.create({})
        for entry in cfg.defaults:
            for group, name in entry.items():
                # climb up to the ``configs/`` root
                config_root = base_dir
                while config_root.name != "configs" and config_root.parent != config_root:
                    config_root = config_root.parent
                sub = OmegaConf.load(config_root / group / f"{name}.yaml")
                merged = OmegaConf.merge(merged, {group: sub})
        cfg = OmegaConf.merge(merged, {k: v for k, v in cfg.items()
                                       if k != "defaults"})
    return cfg


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
