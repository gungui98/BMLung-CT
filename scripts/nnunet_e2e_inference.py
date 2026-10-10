"""End-to-end pipeline: read pre-saved nnU-Net 2D test predictions, run CC
+ per-lesion classifier (3 backbones), write per-case summary JSON.

Streams cases one at a time to avoid OOM on the full test set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import ndimage as ndi
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmct.data.preprocessing import (  # noqa: E402
    PreprocConfig, apply_windowing, crop_or_pad_axial,
    load_ct_volume, resample_isotropic,
)
from bmct.data.schema import load_case_table  # noqa: E402
from bmct.models.nodule_classifier import NoduleClassifier  # noqa: E402

# Paths are set from the command line in main(); see --help.
DATA = None
PREDS = None
RUNS = None
OUT = None
SEED = 2026

CLASSIFIERS = [
    ("resnet18",    "configs/experiment/kc_cancer_nodule_clf.yaml",
                    "kc_cancer_nodule_clf"),
    ("resnet50",    "configs/experiment/kc_cancer_nodule_clf_resnet50.yaml",
                    "kc_cancer_nodule_clf_resnet50"),
    ("densenet121", "configs/experiment/kc_cancer_nodule_clf_densenet121.yaml",
                    "kc_cancer_nodule_clf_densenet121"),
]


def load_cfg(rel: str):
    p = ROOT / rel
    cfg = OmegaConf.load(p)
    cfg_dir = p.parent.parent
    if "defaults" in cfg:
        defaults = cfg.pop("defaults")
        merged = OmegaConf.create({})
        for d in defaults:
            for k, v in d.items():
                merged[k] = OmegaConf.load(cfg_dir / k / f"{v}.yaml")
        cfg = OmegaConf.merge(merged, cfg)
    return cfg


def _to_zyx(arr_xyz):
    return np.transpose(arr_xyz, (2, 1, 0))


def preprocess_ct(nifti_path, preproc):
    vol, sp = load_ct_volume(nifti_path)
    vol_r, _ = resample_isotropic(vol, sp,
                                   preproc.in_plane_spacing_mm,
                                   preproc.slice_spacing_mm, order=1)
    win = preproc.visible_window or preproc.lung_window
    ch = apply_windowing(vol_r, [win])[0]
    return crop_or_pad_axial(ch, preproc.axial_crop)


def load_pred_mask_resampled(case_id, target_shape):
    p = PREDS / f"{case_id}.nii.gz"
    img = nib.load(str(p))
    arr = img.get_fdata().astype(np.uint8)
    arr = _to_zyx(arr)
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(target_shape, arr.shape)]
    return zoom(arr, factors, order=0).astype(np.uint8)


def find_nodules(mask, min_vox=50):
    labeled, n = ndi.label(mask > 0,
                            structure=ndi.generate_binary_structure(3, 1))
    nodules = []
    if n == 0:
        return nodules
    centroids = ndi.center_of_mass(mask > 0, labeled, range(1, n + 1))
    for i in range(1, n + 1):
        comp = labeled == i
        vox = int(comp.sum())
        if vox < min_vox:
            continue
        cz, cy, cx = centroids[i - 1]
        nodules.append({"label": i, "vox": vox,
                         "centroid": (float(cz), float(cy), float(cx))})
    return nodules


def crop_nodule(volume, centroid, crop_px=64):
    Z, Y, X = volume.shape
    cz, cy, cx = (int(round(v)) for v in centroid)
    half = crop_px // 2
    pad = ((1, 1), (half, half), (half, half))
    vp = np.pad(volume, pad, mode="constant")
    zp = cz + 1; yp = cy + half; xp = cx + half
    return vp[zp - 1: zp + 2,
              yp - half: yp + half,
              xp - half: xp + half].astype(np.float32)


def main():
    global DATA, PREDS, RUNS, OUT
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, help="dataset directory holding clinical.csv")
    ap.add_argument("--pred-dir", required=True, help="nnU-Net test predictions (<case_id>.nii.gz)")
    ap.add_argument("--runs-dir", default="runs", help="directory with the trained classifier runs")
    ap.add_argument("--out-dir", default="runs/e2e_combos", help="where per-case JSON files are written")
    args = ap.parse_args()
    DATA, PREDS = Path(args.dataset_root), Path(args.pred_dir)
    RUNS, OUT = Path(args.runs_dir), Path(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] device={device}")

    cfg = load_cfg(CLASSIFIERS[0][1])
    pp = cfg.data.preprocess
    preproc = PreprocConfig(
        in_plane_spacing_mm=float(pp.in_plane_spacing_mm),
        slice_spacing_mm=float(pp.slice_spacing_mm),
        lung_window=tuple(pp.lung_window),
        mediastinal_window=tuple(pp.mediastinal_window),
        visible_window=tuple(pp.visible_window) if getattr(pp, "visible_window", None) else None,
        stack_size=int(getattr(pp, "stack_size", 1)),
        axial_crop=int(pp.axial_crop),
    )

    cases = load_case_table(str(DATA / cfg.data.manifest))
    cases = [c for c in cases if c.split == "test"]
    print(f"[run] n_test_cases={len(cases)}")

    # Load all classifiers once.
    models = {}
    for backbone, cfg_rel, ckpt_rel in CLASSIFIERS:
        c2 = load_cfg(cfg_rel)
        model = NoduleClassifier(c2.model.params).to(device)
        ckpt = torch.load(RUNS / f"{ckpt_rel}_seed{SEED}" / "best.pt", map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt.get("model", ckpt)),
                              strict=False)
        model.eval()
        models[backbone] = model

    # Prepare output dirs + summary buffers
    summaries = {b: [] for b in models}
    out_dirs = {}
    for b in models:
        d = OUT / f"seg-nnunet_cls-{b}_seed{SEED}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[b] = d

    for case in tqdm(cases, desc="cases"):
        try:
            v = preprocess_ct(str(DATA / case.series_path), preproc)
        except Exception as e:
            print(f"[skip] {case.case_id}: {e!r}")
            continue
        # If nnU-Net didn't emit a prediction (e.g. healthy controls), treat
        # as an empty mask -> zero nodules -> case_cancer_prob = 0.
        if not (PREDS / f"{case.case_id}.nii.gz").exists():
            m = np.zeros_like(v, dtype=np.uint8)
        else:
            m = load_pred_mask_resampled(case.case_id, v.shape)
        nodules = find_nodules(m, min_vox=50)
        crops = None
        if nodules:
            crops = np.stack([crop_nodule(v, n["centroid"]) for n in nodules])
            crops_t = torch.from_numpy(crops).to(device)
        for backbone, model in models.items():
            case_p = 0.0
            recs = []
            if nodules:
                with torch.no_grad():
                    out = model(crops_t)
                    cans = torch.sigmoid(out["cancer"]).cpu().numpy()
                for n, k in zip(nodules, cans):
                    n2 = dict(n); n2["cancer_prob"] = float(k)
                    recs.append(n2)
                case_p = float(max(cans))
            summaries[backbone].append({
                "case_id": case.case_id, "gt_label": int(case.label),
                "n_nodules": len(nodules), "case_cancer_prob": case_p,
            })
            (out_dirs[backbone] / f"{case.case_id}.json").write_text(json.dumps({
                "case_id": case.case_id, "gt_label": int(case.label),
                "n_nodules": len(nodules), "case_cancer_prob": case_p,
                "nodules": recs,
            }, indent=2))
        # release per-case memory explicitly
        del v, m
        if crops is not None:
            del crops, crops_t

    from sklearn.metrics import roc_auc_score
    for b, summ in summaries.items():
        (out_dirs[b] / "summary.json").write_text(json.dumps(summ, indent=2))
        y = np.array([s["gt_label"] for s in summ])
        p = np.array([s["case_cancer_prob"] for s in summ])
        print(f"[done] nnunet+{b}: n={len(y)} AUC={roc_auc_score(y, p):.4f}")


if __name__ == "__main__":
    main()
