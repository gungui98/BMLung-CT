"""Batch samplers that balance masked multitask supervision.

The internal cohort contains 2,000 cases; only ~200 carry strong (lesion-level)
annotation.  ``MaskedMultitaskSampler`` guarantees that at least
``min_strong_per_batch`` strongly-annotated cases appear in every minibatch so
that the lesion-segmentation and property losses receive a non-zero gradient.
"""
from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler

from bmct.data.schema import CaseRecord


class MaskedMultitaskSampler(Sampler[List[int]]):
    def __init__(
        self,
        cases: Sequence[CaseRecord],
        batch_size: int,
        min_strong_per_batch: int = 2,
        seed: int = 2026,
        shuffle: bool = True,
        strong_case_ids: Optional[Iterable[str]] = None,
    ) -> None:
        """Balanced batch sampler.

        Strong-annotation membership is supplied explicitly via
        ``strong_case_ids`` (typically the set of keys from the lesion table).
        This keeps the sampler decoupled from any boolean column on
        ``CaseRecord``.
        """
        self.cases = list(cases)
        self.batch_size = batch_size
        self.min_strong = min_strong_per_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        strong_ids = set(strong_case_ids or [])
        self.strong_idx = [i for i, r in enumerate(self.cases)
                           if r.case_id in strong_ids]
        self.weak_idx   = [i for i, r in enumerate(self.cases)
                           if r.case_id not in strong_ids]
        if min_strong_per_batch >= batch_size:
            raise ValueError("min_strong_per_batch must be < batch_size")
        if not self.strong_idx:
            raise ValueError("No strongly-annotated cases found in the split.")

    def __len__(self) -> int:
        return (len(self.cases) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        strong = list(self.strong_idx)
        weak   = list(self.weak_idx)
        if self.shuffle:
            self.rng.shuffle(strong)
            self.rng.shuffle(weak)

        n_batches = len(self)
        s_ptr = w_ptr = 0
        for _ in range(n_batches):
            batch: List[int] = []
            # fill the strong quota (with wrap-around if exhausted)
            for _ in range(self.min_strong):
                if s_ptr >= len(strong):
                    if self.shuffle:
                        self.rng.shuffle(strong)
                    s_ptr = 0
                batch.append(strong[s_ptr]); s_ptr += 1
            # fill the rest with weak samples
            while len(batch) < self.batch_size and w_ptr < len(weak):
                batch.append(weak[w_ptr]); w_ptr += 1
            # if weak pool ran out, top-up with strong (still valid)
            while len(batch) < self.batch_size:
                if s_ptr >= len(strong):
                    if self.shuffle:
                        self.rng.shuffle(strong)
                    s_ptr = 0
                batch.append(strong[s_ptr]); s_ptr += 1
            yield batch
