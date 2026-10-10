"""Cohort splitting for the flat-layout BMCT dataset.

There are two ways a case is assigned to a fold:

1. **Explicit** — ``CaseRecord.split`` is set to ``train`` / ``val`` / ``test``
   in ``clinical.csv``.  This wins.
2. **Derived** — ``CaseRecord.split`` is empty.  Internal cases
   (``cohort == 'bm_internal'``) fall back to the year-range policy defined
   in the experiment YAML (``data.split.{train,validation,test}``).  External
   cases (``multisite`` / ``provincial``) with no explicit split land in
   their own named partitions and are *not* pulled into internal train/val/test.

This keeps the "store the truth once" principle: a researcher can either
freeze the splits explicitly in the CSV, or let the code derive them from
``study_date`` — but never both for the same case.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence

import pandas as pd

from bmct.data.schema import CaseRecord, load_case_table


@dataclass
class Split:
    name: str
    records: List[CaseRecord]


def _parse_year(s: str) -> int:
    return date.fromisoformat(s).year


# ---------------------------------------------------------------------------
# Internal-cohort splits
# ---------------------------------------------------------------------------
def build_splits(
    cases: Sequence[CaseRecord],
    *,
    train_years: Optional[Sequence[int]] = None,
    val_years:   Optional[Sequence[int]] = None,
    test_years:  Optional[Sequence[int]] = None,
    internal_cohort: str = "bm_internal",
) -> Dict[str, Split]:
    """Partition internal cases into ``train`` / ``val`` / ``test``.

    Explicit ``CaseRecord.split`` values take precedence.  Cases without a
    split fall back to year-range windows ``[start, end)`` — the same
    inclusive-exclusive convention used in ``configs/data/*.yaml``.
    """
    if any(r is None for r in (train_years, val_years, test_years)):
        # Derivation requires *all three* ranges; otherwise we simply ignore
        # cases with empty splits in the internal cohort.
        train_years = train_years or []
        val_years   = val_years   or []
        test_years  = test_years  or []

    def within(y: int, rng: Sequence[int]) -> bool:
        return len(rng) == 2 and rng[0] <= y < rng[1]

    buckets: Dict[str, List[CaseRecord]] = {"train": [], "val": [], "test": []}
    for r in cases:
        # If split is explicit, honour it regardless of cohort —
        # new dataset uses cohort names like nhom_benh / nhom_chung
        # that no longer match the legacy bm_internal whitelist.
        if r.split in buckets:
            buckets[r.split].append(r)
            continue
        if r.cohort != internal_cohort:
            continue
        y = _parse_year(r.study_date)
        if within(y, train_years):
            buckets["train"].append(r)
        elif within(y, val_years):
            buckets["val"].append(r)
        elif within(y, test_years):
            buckets["test"].append(r)
        # else: dropped — the user asked for neither explicit split nor a
        # matching year range.  Reported via summarise().
    return {k: Split(k, v) for k, v in buckets.items()}


# Backwards-compatible alias so the old name keeps working.
build_temporal_splits = build_splits


# ---------------------------------------------------------------------------
# External partitions derived directly from the unified clinical.csv
# ---------------------------------------------------------------------------
def build_external_partitions(
    cases: Sequence[CaseRecord],
    cohorts: Sequence[str] = ("multisite", "provincial"),
) -> Dict[str, Split]:
    """Group non-internal cases by ``cohort`` (default: multisite, provincial)."""
    out: Dict[str, Split] = {}
    for cohort in cohorts:
        name = f"external_{cohort}"
        recs = [r for r in cases if r.cohort == cohort]
        out[name] = Split(name=name, records=recs)
    return out


def build_external_partitions_from_cfg(
    cfg_external: Dict[str, dict],
    dataset_root: str,
) -> Dict[str, Split]:
    """Legacy path — load external cases from per-partition manifest CSVs.

    Retained for configs that still point at a manifest file rather than at
    the unified clinical.csv.  New code should prefer
    :func:`build_external_partitions`.
    """
    out: Dict[str, Split] = {}
    for name, spec in cfg_external.items():
        manifest = f"{dataset_root}/{spec['manifest']}"
        out[name] = Split(name=name, records=load_case_table(manifest))
    return out


# ---------------------------------------------------------------------------
# Reference-standard subsets (paper agent's Part D.1 case_level cohorts)
# ---------------------------------------------------------------------------
def filter_records(
    split: Split,
    predicate,
    *,
    name: Optional[str] = None,
) -> Split:
    """Return a new Split containing only records where ``predicate(r)`` is True."""
    recs = [r for r in split.records if predicate(r)]
    return Split(name=name or split.name, records=recs)


def biopsy_subset(split: Split, *, label_source_key: str = "pathology") -> Split:
    """Records with biopsy-confirmed labels (``label_source == 'pathology'``).

    On kc-cancer-final this is the n=164 test subset that the paper's
    case-level malignancy endpoint is reported on as primary.
    """
    def _has_pathology(r: CaseRecord) -> bool:
        return str(r.label_provenance.get("label_source", "")) == label_source_key
    return filter_records(split, _has_pathology, name=f"{split.name}_biopsy")


def consensus_subset(split: Split) -> Split:
    """Records flagged as multi-annotator lesion-annotation consensus
    (``meta.consensus == 'consensus'``).

    On kc-cancer-final this is the n=230 test subset (out of 252; remaining
    22 are 'disagreement'). This is an *annotation-agreement* axis, NOT the
    label-provenance hierarchy — for the paper's case-level secondary cohort
    use :func:`effective_label_source_subset` instead.
    """
    def _is_consensus(r: CaseRecord) -> bool:
        return str(r.meta.get("consensus", "")) == "consensus"
    return filter_records(split, _is_consensus, name=f"{split.name}_consensus")


def effective_label_source_subset(
    split: Split,
    source: str,
) -> Split:
    """Records matching ``meta.effective_label_source == source``.

    kc-cancer-final test breakdown (n=252):
        biopsy                 → 164  (gold standard, primary)
        cohort_annotator_agree →  79  (secondary: cohort label + annotator agreement)
        cohort_only            →   9  (weakest tier)
    """
    def _has_src(r: CaseRecord) -> bool:
        return str(r.meta.get("effective_label_source", "")) == source
    return filter_records(split, _has_src,
                          name=f"{split.name}_{source}")


# ---------------------------------------------------------------------------
def summarise(splits: Dict[str, Split]) -> pd.DataFrame:
    rows = []
    for name, s in splits.items():
        labels = [r.label for r in s.records]
        rows.append({
            "split":    name,
            "n":        len(s.records),
            "malignant": sum(labels),
            "benign":   len(labels) - sum(labels),
        })
    return pd.DataFrame(rows)
