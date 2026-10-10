#!/usr/bin/env python
"""Per-lesion classifier evaluation on a chosen split (default: test).

Reports macro-F1 for density/margin/cavitation and AUC for per-lesion cancer,
on the test crops with GT lesion boxes.
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
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from bmct.data.nodule_dataset import (                              # noqa: E402
    DENSITY_CLASSES, MARGIN_CLASSES,
    NoduleDataset, nodule_collate,
)
from bmct.data.preprocessing import PreprocConfig                    # noqa: E402
from bmct.data.schema import load_case_table, load_lesion_table      # noqa: E402
from bmct.models.nodule_classifier import NoduleClassifier           # noqa: E402


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


def _load_cfg(path: str):
    cfg_path = Path(path)
    cfg = OmegaConf.load(cfg_path)
    cfg_dir = cfg_path.parent.parent
    if "defaults" in cfg:
        defaults = cfg.pop("defaults")
        merged = OmegaConf.create({})
        for d in defaults:
            for k, v in d.items():
                merged[k] = OmegaConf.load(cfg_dir / k / f"{v}.yaml")
        cfg = OmegaConf.merge(merged, cfg)
    return cfg


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    P_dens, P_margin, P_cavit_p, P_cancer_p = [], [], [], []
    T_dens, T_margin, T_cavit, T_cancer = [], [], [], []
    case_ids, lesion_ids = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.image)
        P_dens.append(out["density"].argmax(-1).cpu().numpy())
        P_margin.append(out["margin"].argmax(-1).cpu().numpy())
        P_cavit_p.append(torch.sigmoid(out["cavit"]).cpu().numpy())
        P_cancer_p.append(torch.sigmoid(out["cancer"]).cpu().numpy())
        T_dens.append(batch.density.cpu().numpy())
        T_margin.append(batch.margin.cpu().numpy())
        T_cavit.append(batch.cavit.cpu().numpy())
        T_cancer.append(batch.cancer.cpu().numpy())
        case_ids.extend(list(batch.case_id))
        lesion_ids.extend(list(batch.lesion_id))
    P_dens = np.concatenate(P_dens); T_dens = np.concatenate(T_dens)
    P_margin = np.concatenate(P_margin); T_margin = np.concatenate(T_margin)
    P_cavit_p = np.concatenate(P_cavit_p); T_cavit = np.concatenate(T_cavit)
    P_cancer_p = np.concatenate(P_cancer_p); T_cancer = np.concatenate(T_cancer)
    P_cavit = (P_cavit_p > 0.5).astype(np.int32)

    def macro_f1(P, T, n_classes):
        keep = T >= 0
        if keep.sum() == 0: return float("nan")
        P, T = P[keep], T[keep]
        f1s = []
        for k in range(n_classes):
            tp = ((P == k) & (T == k)).sum()
            fp = ((P == k) & (T != k)).sum()
            fn = ((P != k) & (T == k)).sum()
            denom = 2 * tp + fp + fn
            f1s.append(0.0 if denom == 0 else float(2 * tp / denom))
        return float(np.mean(f1s))

    def acc(P, T):
        keep = T >= 0 if T.dtype != np.float32 else ~np.isnan(T)
        return float((P[keep] == T[keep]).mean()) if keep.sum() else float("nan")

    def cavit_metrics(P, T):
        keep = ~np.isnan(T)
        if keep.sum() == 0: return float("nan"), float("nan")
        Pk, Tk = P[keep].astype(np.float32), T[keep].astype(np.float32)
        a = float((Pk == Tk).mean())
        tp = float((Pk * Tk).sum()); fp = float((Pk * (1 - Tk)).sum()); fn = float(((1 - Pk) * Tk).sum())
        denom = 2 * tp + fp + fn
        f1 = float("nan") if denom == 0 else 2 * tp / denom
        return a, f1

    def auroc(P_prob, T):
        keep = ~np.isnan(T)
        if keep.sum() < 2 or len(np.unique(T[keep])) < 2: return float("nan")
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(T[keep], P_prob[keep]))

    cav_acc, cav_f1 = cavit_metrics(P_cavit, T_cavit)
    return {
        "n_lesions": int(len(P_dens)),
        "n_cases": int(len(set(case_ids))),
        "density/acc": acc(P_dens, T_dens),
        "density/macro_f1": macro_f1(P_dens, T_dens, len(DENSITY_CLASSES)),
        "margin/acc": acc(P_margin, T_margin),
        "margin/macro_f1": macro_f1(P_margin, T_margin, len(MARGIN_CLASSES)),
        "cavit/acc": cav_acc,
        "cavit/f1": cav_f1,
        "cancer/auc_lesion": auroc(P_cancer_p, T_cancer.astype(np.float32)),
        "cancer/n_pos": int((T_cancer == 1).sum()),
        "cancer/n_neg": int((T_cancer == 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    root = Path(args.dataset_root)
    preproc = build_preproc(cfg.data.preprocess)

    cases = load_case_table(str(root / cfg.data.manifest))
    cases = [c for c in cases if c.split == args.split]
    print(f"[eval] split={args.split} n_cases={len(cases)}")
    lesion_table = load_lesion_table(str(root / cfg.data.strong_manifest))
    cache_dir = str(root / cfg.data.cache_dir)

    ds = NoduleDataset(
        cases, lesion_table=lesion_table,
        dataset_root=str(root), preproc=preproc,
        crop_px=int(cfg.data.get("crop_px", 64)),
        cache_dir=cache_dir, transforms=None, precompute=True,
    )
    print(f"[eval] n_lesions={len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True,
                    collate_fn=nodule_collate)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    model = NoduleClassifier(cfg.model.params).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    metrics = evaluate(model, dl, device)
    print(json.dumps(metrics, indent=2))

    out_json = args.out_json or str(Path(args.checkpoint).parent / f"nodule_eval_{args.split}.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "split": args.split,
                "checkpoint": args.checkpoint, "config": args.config}
    Path(out_json).write_text(json.dumps(payload, indent=2))
    print(f"[eval] wrote {out_json}")


if __name__ == "__main__":
    main()
