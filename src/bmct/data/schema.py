"""Schemas for the Bach-Mai chest-CT cohort (flat layout + CLAIM metadata).

The dataset is described by **two CSV tables** sitting at the dataset root:

    clinical.csv      one row per case.  Sole source of truth for cohort /
                      site / split / label plus all manuscript-required
                      metadata (scanner, acquisition, label provenance,
                      clinical variables).
    lesions.csv       one row per annotated lesion (subset of cases).  Links
                      to its binary NIfTI mask via ``mask_path`` and carries
                      the property-ontology labels **plus** per-lesion
                      annotator / review provenance.

Strong annotation membership is *derived*: a case is strongly annotated iff
its ``case_id`` appears in ``lesions.csv``.  No separate flag column.

The column sets are grouped into semantic bundles so downstream code can
address them one group at a time (e.g., the scanner panel in Table 1
consumes ``SCANNER_COLUMNS`` + ``ACQUISITION_COLUMNS`` only).  Every bundle
is optional on disk — missing columns simply produce empty dicts, so
extending the schema is backward-compatible for earlier cohorts.

Per-partition CSVs under ``manifests/`` are an optional cache produced by
``scripts/prepare_manifests.py``; they are never the source of truth.

NIfTI files live under a single flat ``images/`` folder — see
``dataset/README.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Property ontology: categorical labels attached to each annotated lesion.
# Keep these names stable — they are referenced by the data YAMLs and by the
# property heads in the proposed model.
# ---------------------------------------------------------------------------
PROPERTY_GROUPS: Dict[str, List[str]] = {
    "density":              ["solid", "part_solid", "ground_glass"],
    "margin":               ["smooth", "lobulated", "spiculated", "irregular"],
    "lobe":                 ["RUL", "RML", "RLL", "LUL", "LLL"],
    "inflammatory_pattern": ["peripheral_halo", "consolidation", "tree_in_bud"],
    "cavitation":           ["absent", "present"],
}


# ---------------------------------------------------------------------------
# Column groups on ``clinical.csv``
# ---------------------------------------------------------------------------
CLINICAL_COLUMNS: List[str] = [
    # Core clinical variables (model-consumable).
    "age",
    "sex",                          # {M, F}
    "smoking_status",               # {never, former, current}
    "pack_years",
    "symptom_duration_weeks",
    # Extended clinical (Table 1 only; not fed to the model today).
    "bmi",
    "copd",                         # 0 / 1
    "prior_cancer",                 # 0 / 1
    "family_history_lung_cancer",   # 0 / 1
]

# Scanner provenance (per-unit) — CLAIM item 17.
SCANNER_COLUMNS: List[str] = [
    "scanner_manufacturer",         # {Siemens, GE, Philips, Canon, Toshiba, Other}
    "scanner_model",                # free text (e.g., SOMATOM Definition AS)
    "scanner_id",                   # anonymised per-unit tag (e.g., SCANNER_01)
]

# Acquisition parameters — CLAIM items 17–18.
ACQUISITION_COLUMNS: List[str] = [
    "kvp",                          # int, tube voltage
    "tube_current_ma",              # float, effective mA
    "slice_thickness_mm",           # float
    "pixel_spacing_mm",             # float, in-plane (assumed isotropic x=y)
    "reconstruction_kernel",        # free text (e.g., B30f, LUNG)
    "contrast_phase",               # {non_contrast, arterial, venous, delayed, unknown}
    "iv_contrast",                  # 0 / 1
    "ctdi_vol_mgy",                 # float, CTDIvol in mGy (optional)
    "dlp_mgy_cm",                   # float, DLP in mGy·cm (optional)
]

# Label / reference-standard provenance — CLAIM items 20, 26.
LABEL_PROVENANCE_COLUMNS: List[str] = [
    "label_source",                 # {pathology, clinical_followup, tumor_board,
                                    #  adjudicated_imaging}
    "pathology_date",               # ISO date or empty
    "followup_days",                # int, used when label_source=clinical_followup
]


# ---------------------------------------------------------------------------
# Column groups on ``lesions.csv``
# ---------------------------------------------------------------------------
# Per-lesion annotator / review provenance — CLAIM items 19, 21.
ANNOTATOR_COLUMNS: List[str] = [
    "annotator_id",                 # e.g., ANNOT_01
    "annotator_role",               # {attending_radiologist, subspecialist_thoracic,
                                    #  resident}
    "annotator_years_experience",   # int
    "annotation_date",              # ISO date
    "review_status",                # {initial, reviewed, adjudicated}
    "reviewer_id",                  # empty unless review_status ∈ {reviewed, adjudicated}
]


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------
VALID_COHORTS = {"bm_internal", "multisite", "provincial",
                  "nhom_benh", "nhom_chung", "nc_bo_sung"}
VALID_SPLITS  = {"train", "val", "test"}
VALID_SCANNER_MANUFACTURERS = {"Siemens", "GE", "Philips", "Canon",
                                "Toshiba", "Other"}
VALID_CONTRAST_PHASES = {"non_contrast", "arterial", "venous", "delayed",
                          "unknown"}
VALID_LABEL_SOURCES = {"pathology", "clinical_followup", "tumor_board",
                        "adjudicated_imaging"}
VALID_ANNOTATOR_ROLES = {"attending_radiologist", "subspecialist_thoracic",
                          "resident"}
VALID_REVIEW_STATUSES = {"initial", "reviewed", "adjudicated"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PropertyLabels:
    """Categorical labels for a single annotated lesion."""
    density: Optional[str] = None
    margin: Optional[str] = None
    lobe: Optional[str] = None
    inflammatory_pattern: Optional[str] = None
    cavitation: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


@dataclass
class LesionRecord:
    """One annotator's drawing of one lesion.

    The compound primary key is ``(case_id, lesion_id, annotator_id)`` —
    multiple rows can describe the same anatomical lesion, one per reader,
    which feeds inter-rater reliability (CLAIM item 21).  Exactly one row
    per ``(case_id, lesion_id)`` carries ``is_primary=True``; that row is
    the canonical annotation consumed by the training loader and the
    nnU-Net export.
    """
    case_id: str
    lesion_id: str
    mask_path: str                 # NIfTI path (relative to dataset root)
    long_axis_mm: float
    short_axis_mm: float
    volume_mm3: float
    annotator_id: Optional[str] = None   # None only for single-reader cohorts
    is_primary: bool = True
    properties: PropertyLabels = field(default_factory=PropertyLabels)
    annotator: Dict[str, Any] = field(default_factory=dict)  # role / years / dates / review

    @property
    def size_category(self) -> str:
        d = self.long_axis_mm
        if d < 10.0:
            return "small"
        if d < 20.0:
            return "medium"
        return "large"


@dataclass
class CaseRecord:
    """One chest-CT case, flat-layout edition.

    ``series_path`` points at a NIfTI (``.nii`` / ``.nii.gz``) file relative
    to the dataset root (typically under ``images/``).  ``cohort`` identifies
    the partition family; ``split`` — when set — pins the case to a specific
    fold, otherwise splits.py falls back to year-range logic.

    The ``scanner`` / ``acquisition`` / ``label_provenance`` dicts carry the
    CLAIM-aligned metadata fields (optional; empty if the columns are
    absent in the source CSV).
    """
    case_id: str
    series_path: str                   # NIfTI file (relative to dataset root)
    cohort: str                        # bm_internal | multisite | provincial
    site: str                          # site code (bm_internal for single-site)
    study_date: str                    # ISO8601 yyyy-mm-dd
    label: int                         # 0 benign / 1 malignant
    split: Optional[str] = None        # train | val | test | None (= derive)
    clinical: Dict[str, Any] = field(default_factory=dict)
    scanner: Dict[str, Any] = field(default_factory=dict)
    acquisition: Dict[str, Any] = field(default_factory=dict)
    label_provenance: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- row adapters -------------------------------------------------------
    @classmethod
    def from_row(cls, row: pd.Series) -> "CaseRecord":
        cid = row.get("case_id")

        # split ----------------------------------------------------------
        split = row.get("split")
        if isinstance(split, float) and pd.isna(split):
            split = None
        if split == "":
            split = None
        if split is not None:
            split = str(split)
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"case_id={cid}: split={split!r} is not one "
                    f"of {sorted(VALID_SPLITS)} (or empty to derive)"
                )

        # cohort ---------------------------------------------------------
        cohort_raw = row.get("cohort", "bm_internal")
        cohort = "bm_internal" if (cohort_raw is None or
                                     (isinstance(cohort_raw, float) and pd.isna(cohort_raw))
                                     or str(cohort_raw).lower() in ("nan", "none", ""))\
                                  else str(cohort_raw)
        if cohort not in VALID_COHORTS:
            cohort = "bm_internal"

        # grouped columns (skip NaN/empty so dicts stay clean) ----------
        def _group(cols: List[str]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for c in cols:
                if c not in row:
                    continue
                v = row[c]
                if isinstance(v, float) and pd.isna(v):
                    continue
                if v == "":
                    continue
                out[c] = v
            return out

        clinical         = _group(CLINICAL_COLUMNS)
        scanner          = _group(SCANNER_COLUMNS)
        acquisition      = _group(ACQUISITION_COLUMNS)
        label_provenance = _group(LABEL_PROVENANCE_COLUMNS)

        # token validation ----------------------------------------------
        if "scanner_manufacturer" in scanner:
            mfr = str(scanner["scanner_manufacturer"])
            if mfr not in VALID_SCANNER_MANUFACTURERS:
                # Soft-fold unknown vendor → "Other" instead of failing.
                scanner["scanner_manufacturer"] = "Other"
        if "contrast_phase" in acquisition:
            cp = str(acquisition["contrast_phase"])
            if cp not in VALID_CONTRAST_PHASES:
                raise ValueError(
                    f"case_id={cid}: contrast_phase={cp!r} not in "
                    f"{sorted(VALID_CONTRAST_PHASES)}"
                )
        if "label_source" in label_provenance:
            ls = str(label_provenance["label_source"])
            if ls not in VALID_LABEL_SOURCES:
                raise ValueError(
                    f"case_id={cid}: label_source={ls!r} not in "
                    f"{sorted(VALID_LABEL_SOURCES)}"
                )

        reserved = {"case_id", "series_path", "cohort", "site", "study_date",
                    "label", "split",
                    *CLINICAL_COLUMNS, *SCANNER_COLUMNS,
                    *ACQUISITION_COLUMNS, *LABEL_PROVENANCE_COLUMNS}
        meta = {k: row[k] for k in row.index if k not in reserved}

        return cls(
            case_id=str(cid),
            series_path=str(row["series_path"]),
            cohort=cohort,
            site=str(row.get("site", cohort)),
            study_date=str(row.get("study_date", "")),
            label=int(row["label"]),
            split=split,
            clinical=clinical,
            scanner=scanner,
            acquisition=acquisition,
            label_provenance=label_provenance,
            meta=meta,
        )

    def to_row(self) -> Dict[str, Any]:
        return {
            "case_id":    self.case_id,
            "series_path": self.series_path,
            "cohort":     self.cohort,
            "site":       self.site,
            "study_date": self.study_date,
            "label":      self.label,
            "split":      self.split or "",
            **self.clinical,
            **self.scanner,
            **self.acquisition,
            **self.label_provenance,
            **self.meta,
        }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_case_table(path: str) -> List[CaseRecord]:
    """Load the unified cases CSV (``clinical.csv``) into CaseRecords."""
    df = pd.read_csv(path)
    required = {"case_id", "series_path", "cohort", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return [CaseRecord.from_row(r) for _, r in df.iterrows()]


def load_lesion_table(
    path: str,
    *,
    primary_only: bool = True,
) -> Dict[str, List[LesionRecord]]:
    """Load ``lesions.csv`` grouped by ``case_id``.

    Returns an empty dict if the file does not exist (weak-label-only cohorts).

    The compound PK on ``lesions.csv`` is ``(case_id, lesion_id, annotator_id)``.
    When ``primary_only=True`` (the default, suitable for training / nnU-Net
    export) the returned table contains exactly one row per ``(case_id,
    lesion_id)`` — the row whose ``is_primary`` column is truthy.  When
    ``primary_only=False`` all rows are returned; callers doing inter-rater
    agreement analyses can group by ``(case_id, lesion_id)`` and iterate over
    annotators.

    Backward compat: if ``is_primary`` is absent on the CSV every row is
    treated as primary.  Compound-key uniqueness and exactly-one-primary
    per lesion are enforced when ``is_primary`` is present.
    """
    from pathlib import Path as _Path
    if not _Path(path).exists():
        return {}
    df = pd.read_csv(path)
    has_primary_col = "is_primary" in df.columns

    # ---- integrity checks ------------------------------------------------
    if "annotator_id" in df.columns and has_primary_col:
        # full multi-reader schema: enforce compound-key uniqueness +
        # exactly-one primary per (case_id, lesion_id)
        key = df[["case_id", "lesion_id", "annotator_id"]].astype(str)
        dup = key.duplicated(keep=False)
        if dup.any():
            sample = key[dup].head(5).to_dict(orient="records")
            raise ValueError(
                f"{path}: duplicate (case_id, lesion_id, annotator_id) rows: "
                f"{sample}"
            )
        # is_primary count may be 0 (unset), 1 (single annotator), or
        # 2 (test-set multi-annotator). All acceptable.

    grouped: Dict[str, List[LesionRecord]] = {}
    for _, r in df.iterrows():
        props = PropertyLabels(**{k: r.get(k) for k in PROPERTY_GROUPS
                                   if k in r and pd.notna(r.get(k))})

        annotator: Dict[str, Any] = {}
        for c in ANNOTATOR_COLUMNS:
            if c in r:
                v = r.get(c)
                if isinstance(v, float) and pd.isna(v):
                    continue
                if v == "":
                    continue
                annotator[c] = v

        # token validation -------------------------------------------------
        if "annotator_role" in annotator:
            role = str(annotator["annotator_role"])
            if role not in VALID_ANNOTATOR_ROLES:
                raise ValueError(
                    f"case_id={r.get('case_id')} lesion={r.get('lesion_id')}: "
                    f"annotator_role={role!r} not in "
                    f"{sorted(VALID_ANNOTATOR_ROLES)}"
                )
        if "review_status" in annotator:
            st = str(annotator["review_status"])
            if st not in VALID_REVIEW_STATUSES:
                raise ValueError(
                    f"case_id={r.get('case_id')} lesion={r.get('lesion_id')}: "
                    f"review_status={st!r} not in "
                    f"{sorted(VALID_REVIEW_STATUSES)}"
                )

        annotator_id = None
        if "annotator_id" in r and pd.notna(r["annotator_id"]) and r["annotator_id"] != "":
            annotator_id = str(r["annotator_id"])

        if has_primary_col:
            raw = r.get("is_primary")
            if isinstance(raw, float) and pd.isna(raw):
                is_primary = False
            else:
                try:
                    is_primary = bool(int(raw))
                except (TypeError, ValueError):
                    is_primary = str(raw).strip().lower() in {"1", "true", "yes", "y"}
        else:
            is_primary = True

        if primary_only and not is_primary:
            continue

        rec = LesionRecord(
            case_id=str(r["case_id"]),
            lesion_id=str(r["lesion_id"]),
            mask_path=str(r["mask_path"]),
            long_axis_mm=float(r["long_axis_mm"]),
            short_axis_mm=float(r["short_axis_mm"]),
            volume_mm3=float(r["volume_mm3"]),
            annotator_id=annotator_id,
            is_primary=is_primary,
            properties=props,
            annotator=annotator,
        )
        grouped.setdefault(rec.case_id, []).append(rec)
    return grouped


def group_readers(
    lesion_table: Dict[str, List[LesionRecord]],
) -> Dict[str, Dict[str, List[LesionRecord]]]:
    """Reshape a lesion table into ``{case_id: {lesion_id: [reader_rows]}}``.

    Typical input is ``load_lesion_table(path, primary_only=False)`` for an
    inter-rater analysis; typical consumer groups the inner list by
    ``annotator_id`` to compute pairwise Dice / Fleiss' kappa / volume
    concordance.
    """
    out: Dict[str, Dict[str, List[LesionRecord]]] = {}
    for cid, recs in lesion_table.items():
        per_lesion: Dict[str, List[LesionRecord]] = {}
        for r in recs:
            per_lesion.setdefault(r.lesion_id, []).append(r)
        out[cid] = per_lesion
    return out


def strong_case_ids(lesion_table: Dict[str, List[LesionRecord]]) -> set:
    """Convenience: set of case_ids that have any annotated lesion."""
    return set(lesion_table.keys())
