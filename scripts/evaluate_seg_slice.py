#!/usr/bin/env python
"""Per-slice segmentation evaluation for UNet2DSlice / UNet2DSliceV2 on
the test split. Mirrors the validation loop in train_slice.py.

Usage:
    python scripts/evaluate_seg_slice.py \\
        --config configs/experiment/kc_cancer_unet2d_slice.yaml \\
        --checkpoint runs/kc_cancer_unet2d_slice_seed2026/best.pt \\
        --dataset-root /vol/.../raw_data/kc-cancer-final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from bmct.data import BMSliceDataset, slice_collate                  # noqa: E402
from bmct.data.preprocessing import PreprocConfig                    # noqa: E402
from bmct.data.schema import load_case_table, load_lesion_table      # noqa: E402
from bmct.models.unet2d_slice import UNet2DSlice                     # noqa: E402
from bmct.models.unet2d_slice_v2 import UNet2DSliceV2                # noqa: E402
from bmct.models.smp_wrappers import UNetPlusPlus2D, DeepLabV3Plus2D # noqa: E402

MODEL_REGISTRY = {
    "bmct.models.unet2d_slice.UNet2DSlice": UNet2DSlice,
    "bmct.models.unet2d_slice_v2.UNet2DSliceV2": UNet2DSliceV2,
    "bmct.models.smp_wrappers.UNetPlusPlus2D": UNetPlusPlus2D,
    "bmct.models.smp_wrappers.DeepLabV3Plus2D": DeepLabV3Plus2D,
}


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


def _slice_dice(logits: torch.Tensor, target: torch.Tensor,
                 thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > thr).float()
    t = (target > 0.5).float()
    dims = (1, 2, 3)
    inter = (pred * t).sum(dim=dims)
    denom = pred.sum(dim=dims) + t.sum(dim=dims)
    return (2 * inter + eps) / (denom + eps)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices_pos = []
    dices_neg = []
    per_case = {}  # case_id -> {pos_dices: [], neg_dices: []}
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.image)["seg_logits"]
        d = _slice_dice(logits, batch.seg_target).cpu().numpy()
        has = batch.seg_target.flatten(1).sum(-1).cpu().numpy() > 0
        for i, cid in enumerate(batch.case_id):
            rec = per_case.setdefault(cid, {"pos": [], "neg": []})
            if has[i]:
                dices_pos.append(float(d[i]))
                rec["pos"].append(float(d[i]))
            else:
                dices_neg.append(float(d[i]))
                rec["neg"].append(float(d[i]))
    case_pos_dice = {
        cid: float(np.mean(v["pos"])) for cid, v in per_case.items() if v["pos"]
    }
    return {
        "n_slices_pos": len(dices_pos),
        "n_slices_neg": len(dices_neg),
        "slice_dice_pos_mean": float(np.mean(dices_pos)) if dices_pos else float("nan"),
        "slice_dice_pos_median": float(np.median(dices_pos)) if dices_pos else float("nan"),
        "slice_dice_pos_std": float(np.std(dices_pos)) if dices_pos else float("nan"),
        "slice_dice_neg_mean": float(np.mean(dices_neg)) if dices_neg else float("nan"),
        "n_cases_pos": len(case_pos_dice),
        "case_pos_dice_mean": float(np.mean(list(case_pos_dice.values()))) if case_pos_dice else float("nan"),
        "case_pos_dice_median": float(np.median(list(case_pos_dice.values()))) if case_pos_dice else float("nan"),
        "case_pos_dice_std": float(np.std(list(case_pos_dice.values()))) if case_pos_dice else float("nan"),
    }, case_pos_dice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    root = Path(args.dataset_root)
    preproc = build_preproc(cfg.data.preprocess)
    stack_2_5d = int(cfg.data.get("stack_2_5d", 1))

    cases = load_case_table(str(root / cfg.data.manifest))
    cases = [c for c in cases if c.split == args.split]
    print(f"[eval] split={args.split} n_cases={len(cases)}")

    lesion_table = load_lesion_table(str(root / cfg.data.strong_manifest))
    cache_dir = str(root / cfg.data.cache_dir)
    ds = BMSliceDataset(
        cases, dataset_root=str(root), preproc=preproc,
        lesion_table=lesion_table, cache_dir=cache_dir, precompute=True,
        transforms=None, stack_2_5d=stack_2_5d,
    )
    print(f"[eval] n_slices={len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True,
                    collate_fn=slice_collate)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    target = str(cfg.model.target)
    if target not in MODEL_REGISTRY:
        raise KeyError(f"Unsupported model target for slice eval: {target}")
    model = MODEL_REGISTRY[target](cfg.model.params).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    metrics, case_pos_dice = evaluate(model, dl, device)
    print(json.dumps(metrics, indent=2))

    out_json = args.out_json
    if out_json is None:
        out_json = str(Path(args.checkpoint).parent / f"seg_eval_{args.split}.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "per_case_pos_dice": case_pos_dice,
                "split": args.split, "checkpoint": args.checkpoint,
                "config": args.config}
    Path(out_json).write_text(json.dumps(payload, indent=2))
    print(f"[eval] wrote {out_json}")


if __name__ == "__main__":
    main()
