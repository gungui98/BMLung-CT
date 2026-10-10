"""Compute boundary-tolerant metrics (Surface Dice @τmm, HD95) on a
prediction folder vs GT folder.

Surface Dice @ τ: fraction of boundary voxels (in pred OR gt) that lie within
τ mm of the *other* mask's boundary. Robust to small label-edge noise.
HD95: 95-th percentile Hausdorff distance between predicted and GT boundaries.

Usage:
    python boundary_metrics.py --pred-dir P --gt-dir G --out-dir O \
        --tolerance-mm 1 2 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm.auto import tqdm


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Voxels on the boundary of ``mask`` (1-voxel-thick shell)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def _crop_to_union(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop both masks to the bounding box of their union.

    Exact for the distances we need: every boundary voxel of either mask lies
    inside the union box, and the Euclidean distance between two voxels inside
    the box does not depend on how much empty space surrounds it. On these
    volumes the lesion occupies a tiny corner of a 352x352xZ array, so this
    turns each distance transform from minutes into milliseconds.
    """
    union = pred | gt
    if not union.any():
        return pred, gt
    sl = ndi.find_objects(union.astype(np.uint8))[0]
    return pred[sl], gt[sl]


def _surface_distances(pred: np.ndarray, gt: np.ndarray,
                        spacing) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred_to_gt_dist, gt_to_pred_dist) in mm, evaluated on each
    side's boundary."""
    pred, gt = _crop_to_union(pred, gt)
    pb = _boundary(pred)
    gb = _boundary(gt)
    if not gb.any() or not pb.any():
        # Degenerate; large penalty.
        return np.array([]), np.array([])
    # Distance from every voxel to the GT boundary surface (mm).
    dt_gt = ndi.distance_transform_edt(~gb, sampling=spacing)
    dt_pr = ndi.distance_transform_edt(~pb, sampling=spacing)
    return dt_gt[pb], dt_pr[gb]


def surface_dice(pred: np.ndarray, gt: np.ndarray,
                  spacing, tol_mm: float) -> float:
    d_p2g, d_g2p = _surface_distances(pred, gt, spacing)
    if len(d_p2g) == 0 or len(d_g2p) == 0:
        return 0.0 if (pred.any() or gt.any()) else 1.0
    inside = int((d_p2g <= tol_mm).sum()) + int((d_g2p <= tol_mm).sum())
    total = len(d_p2g) + len(d_g2p)
    return inside / total if total else 0.0


def hd95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    d_p2g, d_g2p = _surface_distances(pred, gt, spacing)
    if len(d_p2g) == 0 or len(d_g2p) == 0:
        return float("inf")
    return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))


def regular_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool); g = gt.astype(bool)
    inter = int((p & g).sum())
    s = int(p.sum()) + int(g.sum())
    return 2 * inter / s if s else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tolerance-mm", nargs="+", type=float, default=[1, 2, 3])
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir); gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in tqdm(sorted(pred_dir.glob("*.nii.gz")), desc="cases"):
        gt_f = gt_dir / f.name
        if not gt_f.exists():
            continue
        pred_img = nib.load(str(f))
        gt_img   = nib.load(str(gt_f))
        pred = (pred_img.get_fdata() > 0.5).astype(bool)
        gt   = (gt_img.get_fdata() > 0.5).astype(bool)
        if pred.shape != gt.shape:
            continue
        # Spacing from header in (x, y, z) mm.
        spacing = tuple(float(s) for s in pred_img.header.get_zooms()[:3])
        case_id = f.stem.replace(".nii", "")
        row = {
            "case_id": case_id,
            "dice": regular_dice(pred, gt),
            "hd95_mm": hd95(pred, gt, spacing),
            "gt_vox": int(gt.sum()), "pred_vox": int(pred.sum()),
            "gt_empty": int(not gt.any()),
            "pred_empty": int(not pred.any()),
        }
        for t in args.tolerance_mm:
            row[f"surface_dice_{int(t)}mm"] = surface_dice(pred, gt, spacing, t)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "boundary_metrics.csv", index=False)

    finite_hd = df["hd95_mm"].replace([float("inf")], np.nan)
    summary = {
        "n_cases": int(len(df)),
        "n_with_gt": int((~df["gt_empty"].astype(bool)).sum()),
        "mean_dice": float(df["dice"].mean()),
        "median_dice": float(df["dice"].median()),
        "mean_hd95_mm_finite": float(finite_hd.mean()),
        "median_hd95_mm_finite": float(finite_hd.median()),
        "n_hd95_inf": int(df["hd95_mm"].replace([float("inf")], np.nan).isna().sum()),
    }
    for t in args.tolerance_mm:
        col = f"surface_dice_{int(t)}mm"
        summary[f"mean_{col}"] = float(df[col].mean())
        summary[f"median_{col}"] = float(df[col].median())

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
