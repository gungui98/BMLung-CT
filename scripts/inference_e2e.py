"""End-to-end inference: seg → connected-components → per-nodule classifier.

For each test case:
    1. Run UNet (boost ckpt) per slice → per-slice probability map.
    2. Threshold + connected-components (3-D, 6-connectivity).
    3. For each component above a min-volume threshold, find centroid,
       crop a 64×64×3 patch, run the nodule classifier.
    4. Write a JSON per case with the per-nodule predictions and a
       case-level malignancy probability = max(per-nodule cancer probs).

Usage:
    python scripts/inference_e2e.py \
        --dataset-root ../raw_data/kc-cancer-final \
        --data-cfg configs/data/kc_cancer.yaml \
        --seg-ckpt runs/kc_cancer_unet2d_slice_boost/best.pt \
        --seg-model-cfg configs/model/unet2d_slice_big.yaml \
        --nodule-ckpt runs/kc_cancer_nodule_clf/best.pt \
        --nodule-model-cfg configs/model/nodule_classifier.yaml \
        --split test \
        --out-dir runs/e2e_inference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import ndimage as ndi

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from bmct.data.nodule_dataset import DENSITY_CLASSES, MARGIN_CLASSES  # noqa: E402
from bmct.data.preprocessing import (                                  # noqa: E402
    PreprocConfig, apply_windowing, crop_or_pad_axial,
    load_ct_volume, resample_isotropic,
)
from bmct.data.schema import load_case_table                           # noqa: E402
from bmct.models.unet2d_slice import UNet2DSlice                       # noqa: E402
from bmct.models.nodule_classifier import NoduleClassifier             # noqa: E402


def build_preproc(p) -> PreprocConfig:
    return PreprocConfig(
        in_plane_spacing_mm=float(p.in_plane_spacing_mm),
        slice_spacing_mm=float(p.slice_spacing_mm),
        lung_window=tuple(p.lung_window),
        mediastinal_window=tuple(p.mediastinal_window),
        visible_window=tuple(p.visible_window) if getattr(p, "visible_window", None) else None,
        stack_size=int(getattr(p, "stack_size", 1)),
        axial_crop=int(p.axial_crop),
    )


def preprocess_case(nifti_path: str, cfg: PreprocConfig) -> np.ndarray:
    """Return (Z, axial_crop, axial_crop) float32 in [0,1]."""
    vol, spacing = load_ct_volume(nifti_path)
    vol_r, _ = resample_isotropic(vol, spacing,
                                   cfg.in_plane_spacing_mm,
                                   cfg.slice_spacing_mm, order=1)
    win = cfg.visible_window or cfg.lung_window
    ch = apply_windowing(vol_r, [win])[0]                  # (Z, Y, X)
    return crop_or_pad_axial(ch, cfg.axial_crop)


@torch.no_grad()
def run_seg(model, volume: np.ndarray, device, batch_size: int = 8,
             stack_2_5d: int = 1) -> np.ndarray:
    """Per-slice forward → (Z, Y, X) probability map.

    When ``stack_2_5d > 1`` (must be odd), each slice is fed with that many
    adjacent slices stacked into the channel dim, matching the V2 training
    pipeline.
    """
    Z, Y, X = volume.shape
    probs = np.zeros((Z, Y, X), dtype=np.float32)
    half = stack_2_5d // 2

    # Build 2.5D stacks per slice index (cheap, no extra mem if stack_2_5d=1).
    def stack_for(z):
        if stack_2_5d == 1:
            return volume[z][None]
        zs = [max(0, min(Z - 1, z + d)) for d in range(-half, half + 1)]
        return np.stack([volume[k] for k in zs], axis=0)

    for s in range(0, Z, batch_size):
        e = min(s + batch_size, Z)
        chunk = np.stack([stack_for(z) for z in range(s, e)], axis=0)
        x = torch.from_numpy(chunk).to(device)
        out = model(x)["seg_logits"]
        probs[s:e] = torch.sigmoid(out).squeeze(1).cpu().numpy()
    return probs


def find_nodules(seg_prob: np.ndarray, thr: float = 0.5,
                  min_vox: int = 50, peak_thr: float = 0.0,
                  morph_close: int = 0) -> List[dict]:
    """3-D connected-components on the thresholded seg map, with optional
    morphological closing and a per-component peak-confidence filter.
    """
    mask = seg_prob > thr
    if morph_close > 0:
        from scipy.ndimage import binary_closing
        mask = binary_closing(mask, iterations=morph_close)
    labeled, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    centroids = ndi.center_of_mass(mask, labeled, range(1, n + 1))
    nodules = []
    for i in range(1, n + 1):
        comp = labeled == i
        vox = int(comp.sum())
        if vox < min_vox:
            continue
        peak = float(seg_prob[comp].max())
        if peak < peak_thr:
            continue
        cz, cy, cx = centroids[i - 1]
        nodules.append({"label": i, "vox": vox, "peak": round(peak, 3),
                         "centroid": (float(cz), float(cy), float(cx))})
    return nodules


def crop_nodule(volume: np.ndarray, centroid, crop_px: int = 64) -> np.ndarray:
    """3-channel crop centred on centroid (z-1, z, z+1)."""
    Z, Y, X = volume.shape
    cz, cy, cx = (int(round(v)) for v in centroid)
    half = crop_px // 2
    pad = ((1, 1), (half, half), (half, half))
    vp = np.pad(volume, pad, mode="constant")
    zp = cz + 1; yp = cy + half; xp = cx + half
    return vp[zp - 1: zp + 2,
              yp - half: yp + half,
              xp - half: xp + half].astype(np.float32)


@torch.no_grad()
def classify_nodules(clf, crops: np.ndarray, device) -> List[dict]:
    x = torch.from_numpy(crops).to(device)
    out = clf(x)
    dp = torch.softmax(out["density"], -1).cpu().numpy()
    mp = torch.softmax(out["margin"], -1).cpu().numpy()
    cp = torch.sigmoid(out["cavit"]).cpu().numpy()
    kp = torch.sigmoid(out["cancer"]).cpu().numpy()
    results = []
    for i in range(len(crops)):
        results.append({
            "density":  {c: round(float(dp[i, k]), 3)
                          for k, c in enumerate(DENSITY_CLASSES)},
            "margin":   {c: round(float(mp[i, k]), 3)
                          for k, c in enumerate(MARGIN_CLASSES)},
            "cavit_prob":  round(float(cp[i]), 3),
            "cancer_prob": round(float(kp[i]), 3),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--data-cfg", required=True)
    ap.add_argument("--seg-ckpt", required=True)
    ap.add_argument("--seg-model-cfg", required=True)
    ap.add_argument("--nodule-ckpt", required=True)
    ap.add_argument("--nodule-model-cfg", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="runs/e2e_inference")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-vox", type=int, default=50)
    ap.add_argument("--peak-thr", type=float, default=0.0,
                    help="filter components whose max prob < this")
    ap.add_argument("--morph-close", type=int, default=0,
                    help="iterations of 3-D binary closing before CC")
    ap.add_argument("--agg", choices=["max", "topk_mean"], default="max")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_cfg = OmegaConf.load(args.data_cfg)
    preproc = build_preproc(data_cfg.preprocess)
    root = Path(args.dataset_root)
    cases = load_case_table(str(root / data_cfg.manifest))
    cases = [c for c in cases if c.split == args.split]
    print(f"[infer] {len(cases)} cases in split={args.split}")

    seg_yaml = OmegaConf.load(args.seg_model_cfg)
    seg_cfg = seg_yaml.params
    target = str(seg_yaml.get("target", "")).lower()
    if "unetplusplus" in target or "smp_wrappers.unetpp" in target:
        from bmct.models.smp_wrappers import UNetPlusPlus2D
        seg = UNetPlusPlus2D(seg_cfg).to(device)
        stack_2_5d = 1
    elif "deeplabv3" in target:
        from bmct.models.smp_wrappers import DeepLabV3Plus2D
        seg = DeepLabV3Plus2D(seg_cfg).to(device)
        stack_2_5d = 1
    elif "v2" in target:
        from bmct.models.unet2d_slice_v2 import UNet2DSliceV2
        seg = UNet2DSliceV2(seg_cfg).to(device)
        stack_2_5d = int(seg_cfg.get("in_channels", 1))
    else:
        seg = UNet2DSlice(seg_cfg).to(device)
        stack_2_5d = 1
    seg.load_state_dict(torch.load(args.seg_ckpt, map_location=device,
                                    weights_only=False)["state_dict"])
    seg.eval()
    print(f"[seg] stack_2_5d={stack_2_5d}")

    nod_cfg = OmegaConf.load(args.nodule_model_cfg).params
    clf = NoduleClassifier(nod_cfg).to(device)
    clf.load_state_dict(torch.load(args.nodule_ckpt, map_location=device,
                                    weights_only=False)["state_dict"])
    clf.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    from tqdm.auto import tqdm
    for case in tqdm(cases, desc="cases"):
        try:
            vol = preprocess_case(str(root / case.series_path), preproc)
        except Exception as e:
            print(f"[skip] {case.case_id}: {e!r}")
            continue
        seg_prob = run_seg(seg, vol, device, stack_2_5d=stack_2_5d)
        nodules = find_nodules(seg_prob, thr=args.thr, min_vox=args.min_vox,
                                peak_thr=args.peak_thr,
                                morph_close=args.morph_close)

        case_report = {
            "case_id": case.case_id,
            "gt_label": int(case.label),
            "n_nodules": len(nodules),
            "nodules": [],
            "case_cancer_prob": 0.0,
        }
        if nodules:
            crops = np.stack([crop_nodule(vol, n["centroid"]) for n in nodules])
            clf_out = classify_nodules(clf, crops, device)
            for n, c in zip(nodules, clf_out):
                n.update(c)
                case_report["nodules"].append(n)
            probs_sorted = sorted(
                (n["cancer_prob"] for n in case_report["nodules"]),
                reverse=True,
            )
            if args.agg == "max":
                case_report["case_cancer_prob"] = float(probs_sorted[0])
            else:
                k = min(args.topk, len(probs_sorted))
                case_report["case_cancer_prob"] = float(
                    sum(probs_sorted[:k]) / k)

        (out_dir / f"{case.case_id}.json").write_text(
            json.dumps(case_report, indent=2))
        summary.append({
            "case_id": case.case_id,
            "gt_label": case_report["gt_label"],
            "n_nodules": case_report["n_nodules"],
            "case_cancer_prob": case_report["case_cancer_prob"],
        })

    sum_path = out_dir / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))

    # Quick case-level AUC.
    try:
        from sklearn.metrics import roc_auc_score
        y = np.array([s["gt_label"] for s in summary])
        p = np.array([s["case_cancer_prob"] for s in summary])
        if len(set(y)) > 1:
            print(f"[case-level] n={len(y)}  AUC={roc_auc_score(y, p):.4f}  "
                  f"prevalence={y.mean():.3f}")
    except Exception as e:
        print(f"[auc] skip: {e!r}")

    print(f"[done] per-case reports in {out_dir}")


if __name__ == "__main__":
    main()
