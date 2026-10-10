"""Deterministic seeding."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 2026, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(False)     # keep fallback for cudnn
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
