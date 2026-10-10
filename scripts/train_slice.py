"""Slice-level U-Net training entry point (segmentation-only smoke test).

Usage:
    python scripts/train_slice.py \
        --dataset-root ../raw_data/kc-cancer-final \
        --config configs/experiment/kc_cancer_unet2d_slice.yaml

The script is intentionally minimal — no MIL, no property heads, no
clinical fusion. It loads one slice per item from :class:`BMSliceDataset`,
runs a 2-D U-Net, and minimises Dice + BCE on the per-slice mask.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from bmct.data import BMSliceDataset, slice_collate                  # noqa: E402
from bmct.data.preprocessing import PreprocConfig                    # noqa: E402
from bmct.data.schema import load_case_table, load_lesion_table      # noqa: E402
from bmct.data.slice_transforms import build_train_paired             # noqa: E402
from bmct.losses.dice import soft_dice_loss                          # noqa: E402
from bmct.losses.tversky import soft_tversky_loss                    # noqa: E402
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


def build_loaders(cfg, dataset_root: Path, limit_cases: int | None = None):
    cases = load_case_table(str(dataset_root / cfg.data.manifest))
    lesion_table = load_lesion_table(str(dataset_root / cfg.data.strong_manifest))
    preproc = build_preproc(cfg.data.preprocess)
    cache_dir = str(dataset_root / cfg.data.cache_dir)

    splits = {s: [c for c in cases if c.split == s] for s in ("train", "val", "test")}
    if limit_cases:
        splits = {s: v[:limit_cases] for s, v in splits.items()}
    print(f"[splits] train={len(splits['train'])}  "
          f"val={len(splits['val'])}  test={len(splits['test'])}")

    aug_cfg = cfg.get("augmentation", None)
    train_tx = None
    if aug_cfg and bool(aug_cfg.get("enabled", False)):
        train_tx = build_train_paired(
            rotate_deg=float(aug_cfg.get("rotate_deg", 15.0)),
            shift_px=int(aug_cfg.get("shift_px", 16)),
            brightness=float(aug_cfg.get("brightness", 0.05)),
            contrast=float(aug_cfg.get("contrast", 0.05)),
        )
        print(f"[aug] enabled: rotate=±{aug_cfg.get('rotate_deg', 15)}° "
              f"shift=±{aug_cfg.get('shift_px', 16)}px")

    stack_2_5d = int(cfg.data.get("stack_2_5d", 1))

    def make(records, shuffle):
        ds = BMSliceDataset(
            records, dataset_root=str(dataset_root), preproc=preproc,
            lesion_table=lesion_table, cache_dir=cache_dir, precompute=True,
            transforms=train_tx if shuffle else None,
            stack_2_5d=stack_2_5d,
        )
        return ds, DataLoader(
            ds, batch_size=int(cfg.train.batch_size), shuffle=shuffle,
            num_workers=int(cfg.train.num_workers), pin_memory=True,
            collate_fn=slice_collate, persistent_workers=cfg.train.num_workers > 0,
        )

    train_ds, train_dl = make(splits["train"], shuffle=True)
    val_ds, val_dl = make(splits["val"], shuffle=False)
    print(f"[slices] train={len(train_ds)}  val={len(val_ds)}")
    return train_dl, val_dl


# ---------------------------------------------------------------------------
def compute_loss(seg_logits, seg_target, dice_w, bce_w,
                  tversky_w: float = 0.0, tversky_alpha: float = 0.3,
                  tversky_beta: float = 0.7):
    dice = soft_dice_loss(seg_logits, seg_target)
    bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target)
    total = dice * dice_w + bce * bce_w
    tv_val = 0.0
    if tversky_w > 0:
        tv = soft_tversky_loss(seg_logits, seg_target,
                                alpha=tversky_alpha, beta=tversky_beta)
        total = total + tversky_w * tv
        tv_val = tv.item()
    return total, dice.item(), bce.item(), tv_val


def slice_dice(seg_logits: torch.Tensor, seg_target: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    """Per-slice Dice; returns (B,) tensor."""
    probs = (torch.sigmoid(seg_logits) > 0.5).float()
    dims = (1, 2, 3)
    num = 2 * (probs * seg_target).sum(dim=dims)
    den = probs.sum(dim=dims) + seg_target.sum(dim=dims) + eps
    return num / den


@torch.no_grad()
def evaluate(model, loader, device,
             n_worst_pos: int = 4, n_best_pos: int = 2, n_worst_neg: int = 2):
    """Compute val Dice metrics + collect failure/success exemplars.

    Returns (metrics, panels) where panels is a list of dicts ready for
    :func:`render_panel`, grouped into:
        worst_pos: lowest-Dice slices that *have* a lesion (model misses)
        best_pos:  highest-Dice positive slices (model gets it right)
        worst_neg: highest-FP slices that *don't* have a lesion (model hallucinates)
    """
    model.eval()
    dices_pos, dices_neg = [], []
    # Streaming top-K via tracked lists.
    worst_pos, best_pos, worst_neg = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.image)["seg_logits"]
        d_vec = slice_dice(logits, batch.seg_target).cpu().numpy()
        has = batch.seg_target.flatten(1).sum(-1).cpu().numpy() > 0
        probs = torch.sigmoid(logits).cpu().numpy()
        imgs = batch.image.cpu().numpy()
        gts = batch.seg_target.cpu().numpy()
        fp_px = ((probs > 0.5) & (gts < 0.5)).sum(axis=(1, 2, 3))

        for i in range(imgs.shape[0]):
            sample = {
                "kind": "pos" if has[i] else "neg",
                "img": imgs[i, 0].copy(),
                "gt":  gts[i, 0].copy(),
                "prob": probs[i, 0].copy(),
                "case_id": batch.case_id[i],
                "slice_idx": int(batch.slice_idx[i]),
                "dice": float(d_vec[i]),
                "fp_px": int(fp_px[i]),
            }
            if has[i]:
                dices_pos.append(sample["dice"])
                worst_pos.append(sample)
                worst_pos.sort(key=lambda s: s["dice"])
                worst_pos[:] = worst_pos[:n_worst_pos]
                best_pos.append(sample)
                best_pos.sort(key=lambda s: -s["dice"])
                best_pos[:] = best_pos[:n_best_pos]
            else:
                dices_neg.append(sample["dice"])
                worst_neg.append(sample)
                worst_neg.sort(key=lambda s: -s["fp_px"])
                worst_neg[:] = worst_neg[:n_worst_neg]

    metrics = {
        "val/dice_pos": float(np.mean(dices_pos)) if dices_pos else float("nan"),
        "val/dice_neg": float(np.mean(dices_neg)) if dices_neg else float("nan"),
        "val/n_pos": len(dices_pos),
        "val/n_neg": len(dices_neg),
    }
    panels = {
        "worst_pos": worst_pos,
        "best_pos":  best_pos,
        "worst_neg": worst_neg,
    }
    return metrics, panels


def _legacy_render_panel(sample: dict, group: str) -> np.ndarray:
    """4-up figure: image | TP/FP/FN error-map | probability heatmap | hist."""
    import matplotlib.pyplot as plt

    img = sample["img"]; gt = sample["gt"]; prob = sample["prob"]
    pred = prob > 0.5
    gt_b = gt > 0.5
    tp = pred & gt_b
    fp = pred & ~gt_b
    fn = ~pred & gt_b

    # Error map: TP=green, FP=red, FN=yellow.
    err = np.stack([img, img, img], axis=-1)
    err[tp] = 0.45 * err[tp] + 0.55 * np.array([0.0, 1.0, 0.0])
    err[fp] = 0.45 * err[fp] + 0.55 * np.array([1.0, 0.0, 0.0])
    err[fn] = 0.45 * err[fn] + 0.55 * np.array([1.0, 1.0, 0.0])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), dpi=90)
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("image")
    axes[1].imshow(err)
    axes[1].set_title(f"err: TP={int(tp.sum())} FP={int(fp.sum())} "
                       f"FN={int(fn.sum())}")
    axes[2].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(prob, cmap="jet", vmin=0, vmax=1, alpha=0.5)
    axes[2].set_title("sigmoid heatmap")
    for ax in axes:
        ax.axis("off")
    title = (f"[{group}] {sample['case_id']} z={sample['slice_idx']}  "
             f"dice={sample['dice']:.3f}  fp_px={sample['fp_px']}")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override cfg.train.epochs")
    ap.add_argument("--limit-train-batches", type=int, default=None,
                    help="cap steps per epoch (smoke)")
    ap.add_argument("--limit-cases", type=int, default=None,
                    help="cap N cases per split (smoke test)")
    ap.add_argument("--device", default=None, help="override device")
    ap.add_argument("--resume", default=None,
                    help="path to checkpoint (best.pt) to warm-start from")
    ap.add_argument("overrides", nargs="*",
                    help="OmegaConf dot-path overrides, e.g. train.seed=2027")
    args = ap.parse_args()

    # Merge experiment config with its defaults.
    cfg_path = Path(args.config)
    cfg = OmegaConf.load(cfg_path)
    cfg_dir = cfg_path.parent.parent
    if "defaults" in cfg:
        defaults = cfg.pop("defaults")
        merged = OmegaConf.create({})
        for d in defaults:
            for k, v in d.items():
                merged[k] = OmegaConf.load(cfg_dir / k / f"{v}.yaml")
        cfg = OmegaConf.merge(merged, cfg)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] output_dir={out_dir}")
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")

    torch.manual_seed(int(cfg.train.seed))
    dev_str = args.device or cfg.train.device
    if dev_str == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable, falling back to cpu")
        dev_str = "cpu"
    device = torch.device(dev_str)

    train_dl, val_dl = build_loaders(cfg, dataset_root,
                                     limit_cases=args.limit_cases)

    model_cls = MODEL_REGISTRY.get(str(cfg.model.target), UNet2DSlice)
    model = model_cls(cfg.model.params).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    extras = []
    for k in ("in_channels", "base_channels", "encoder_name"):
        v = cfg.model.params.get(k, None) if hasattr(cfg.model.params, "get") else None
        if v is not None:
            extras.append(f"{k}={v}")
    print(f"[model] {cfg.model.name}  params={n_params/1e6:.2f}M  "
          + "  ".join(extras))

    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(cfg.train.optimizer.lr),
                            weight_decay=float(cfg.train.optimizer.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg.train.epochs))
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    dice_w = float(cfg.get("loss", {}).get("dice_weight", 1.0))
    bce_w = float(cfg.get("loss", {}).get("bce_weight", 0.5))
    tversky_w = float(cfg.get("loss", {}).get("tversky_weight", 0.0))
    tv_alpha = float(cfg.get("loss", {}).get("tversky_alpha", 0.3))
    tv_beta = float(cfg.get("loss", {}).get("tversky_beta", 0.7))

    # --- resume from checkpoint ----------------------------------------------
    start_epoch = 0
    best_dice = -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        best_dice = float(ck.get("metrics", {}).get("val/dice_pos", -1.0))
        # Advance LR scheduler to the same point.
        for _ in range(start_epoch):
            sched.step()
        print(f"[resume] {args.resume}  start_epoch={start_epoch}  "
              f"best_dice={best_dice:.4f}  lr={opt.param_groups[0]['lr']:.2e}")

    history = []
    log_path = out_dir / "log.jsonl"
    # Resume appends; fresh run truncates.
    log_f = open(log_path, "a" if args.resume else "w")
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    # --- wandb ---------------------------------------------------------------
    wb = None
    log_cfg = OmegaConf.to_container(cfg.train.log, resolve=True) \
        if "log" in cfg.train else {}
    if log_cfg.get("backend") == "wandb":
        try:
            import wandb
            wb = wandb.init(
                project=log_cfg.get("wandb", {}).get("project", "lung-bmct"),
                name=cfg.name,
                dir=str(out_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=log_cfg.get("mode", "online"),
            )
            print(f"[wandb] project={wb.project}  run={wb.name}  url={wb.url}")
        except Exception as e:
            print(f"[wandb] disabled: {e!r}")
            wb = None

    for ep in range(start_epoch, int(cfg.train.epochs)):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "dice": 0.0, "bce": 0.0, "n": 0}
        for step, batch in enumerate(train_dl):
            if args.limit_train_batches and step >= args.limit_train_batches:
                break
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch.image)["seg_logits"]
                loss, d, b, tv = compute_loss(
                    logits, batch.seg_target, dice_w, bce_w,
                    tversky_w=tversky_w,
                    tversky_alpha=tv_alpha, tversky_beta=tv_beta,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            float(cfg.train.grad_clip_norm))
            scaler.step(opt)
            scaler.update()
            bs = batch.image.size(0)
            running["loss"] += loss.item() * bs
            running["dice"] += d * bs
            running["bce"] += b * bs
            running["n"] += bs
        sched.step()
        n = max(running["n"], 1)
        ep_train = {
            "epoch": ep,
            "train/loss": running["loss"] / n,
            "train/dice_loss": running["dice"] / n,
            "train/bce": running["bce"] / n,
            "lr": opt.param_groups[0]["lr"],
            "time_sec": time.time() - t0,
        }
        val, panels = evaluate(model, val_dl, device)
        row = {**ep_train, **val}
        history.append(row)
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        print(f"ep {ep+1:02d}/{int(cfg.train.epochs)}  "
              f"loss={row['train/loss']:.4f} "
              f"dice_loss={row['train/dice_loss']:.4f} "
              f"val_dice_pos={row['val/dice_pos']:.4f} "
              f"val_dice_neg={row['val/dice_neg']:.4f}  "
              f"lr={row['lr']:.2e}  "
              f"{row['time_sec']:.1f}s")

        # --- viz: 3 separate panes via shared module ---------------------
        try:
            from bmct.visualize import log_panels
            log_panels(wb, viz_dir, ep, panels, attr_overlay=False)
        except Exception as e:
            print(f"  [viz] skip: {e!r}")

        if wb is not None:
            wb.log({k: v for k, v in row.items()
                    if isinstance(v, (int, float))}, step=ep)
        if val["val/dice_pos"] > best_dice:
            best_dice = val["val/dice_pos"]
            torch.save({"state_dict": model.state_dict(),
                        "epoch": ep, "metrics": val},
                       out_dir / "best.pt")

    log_f.close()
    if wb is not None:
        wb.finish()
    print(f"\n[done] best val/dice_pos={best_dice:.4f}  log={log_path}")


if __name__ == "__main__":
    main()
