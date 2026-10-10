"""Stage-2 per-nodule classifier training entry point.

One sample = one annotated lesion. Predicts per-lesion density, margin,
cavitation, and per-case cancer/benign label. See
:mod:`bmct.data.nodule_dataset` for label semantics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from bmct.data.nodule_dataset import (                              # noqa: E402
    DENSITY_CLASSES, MARGIN_CLASSES, CAVIT_CLASSES,
    NoduleDataset, nodule_collate,
)
from bmct.data.preprocessing import PreprocConfig                    # noqa: E402
from bmct.data.schema import load_case_table, load_lesion_table      # noqa: E402
from bmct.data.slice_transforms import build_train_paired            # noqa: E402
from bmct.models.nodule_classifier import NoduleClassifier           # noqa: E402


CLASS_VI = {
    "density":  {"solid": "đặc", "part_solid": "bán-đặc",
                  "ground_glass": "kính mờ", "consolidation": "đông đặc"},
    "margin":   {"smooth": "bờ nhẵn", "lobulated": "múi",
                  "spiculated": "tua gai", "irregular": "không đều"},
    "cavitation": {"absent": "không hang", "present": "có hang"},
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


def build_loaders(cfg, dataset_root: Path, limit_cases=None):
    cases = load_case_table(str(dataset_root / cfg.data.manifest))
    lesion_table = load_lesion_table(str(dataset_root / cfg.data.strong_manifest))
    preproc = build_preproc(cfg.data.preprocess)
    cache_dir = str(dataset_root / cfg.data.cache_dir)
    crop_px = int(cfg.data.crop_px)

    splits = {s: [c for c in cases if c.split == s] for s in ("train", "val", "test")}
    if limit_cases:
        splits = {s: v[:limit_cases] for s, v in splits.items()}
    for s, v in splits.items():
        print(f"[cases] {s}: {len(v)}")

    aug_cfg = cfg.get("augmentation", None)
    train_tx = None
    if aug_cfg and bool(aug_cfg.get("enabled", False)):
        train_tx = build_train_paired(
            rotate_deg=float(aug_cfg.get("rotate_deg", 10.0)),
            shift_px=int(aug_cfg.get("shift_px", 4)),
            brightness=float(aug_cfg.get("brightness", 0.05)),
            contrast=float(aug_cfg.get("contrast", 0.05)),
        )
        print(f"[aug] enabled: rotate=±{aug_cfg.get('rotate_deg', 10)}° "
              f"shift=±{aug_cfg.get('shift_px', 4)}px")

    def make(records, shuffle):
        ds = NoduleDataset(
            records, lesion_table,
            dataset_root=str(dataset_root), preproc=preproc,
            cache_dir=cache_dir, crop_px=crop_px,
            transforms=train_tx if shuffle else None,
            precompute=True,
        )
        return ds, DataLoader(
            ds, batch_size=int(cfg.train.batch_size), shuffle=shuffle,
            num_workers=int(cfg.train.num_workers), pin_memory=True,
            collate_fn=nodule_collate,
            persistent_workers=cfg.train.num_workers > 0,
        )

    train_ds, train_dl = make(splits["train"], shuffle=True)
    val_ds, val_dl = make(splits["val"], shuffle=False)
    print(f"[nodules] train={len(train_ds)}  val={len(val_ds)}")
    return train_dl, val_dl


# ---------------------------------------------------------------------------
def masked_ce(logits, target, weight=None):
    keep = target >= 0
    if not keep.any():
        return logits.new_zeros(())
    return F.cross_entropy(logits[keep], target[keep], weight=weight)


def masked_bce(logits, target, pos_weight=None):
    keep = ~torch.isnan(target)
    if not keep.any():
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(
        logits[keep], target[keep], pos_weight=pos_weight)


def compute_loss(out, batch, w, class_w) -> tuple[torch.Tensor, Dict[str, float]]:
    l_dens   = masked_ce(out["density"], batch.density,
                          weight=class_w.get("density"))
    l_margin = masked_ce(out["margin"],  batch.margin,
                          weight=class_w.get("margin"))
    l_cavit  = masked_bce(out["cavit"],  batch.cavit,
                           pos_weight=class_w.get("cavit"))
    l_cancer = masked_bce(out["cancer"], batch.cancer,
                           pos_weight=class_w.get("cancer"))
    total = (w["density_weight"] * l_dens
             + w["margin_weight"] * l_margin
             + w["cavit_weight"] * l_cavit
             + w["cancer_weight"] * l_cancer)
    return total, {"density": l_dens.item(), "margin": l_margin.item(),
                   "cavit": l_cavit.item(), "cancer": l_cancer.item()}


def compute_class_weights(loader, device) -> Dict[str, torch.Tensor]:
    """Inverse-frequency weights from one pass over the train loader."""
    dens_cnt = np.zeros(len(DENSITY_CLASSES))
    marg_cnt = np.zeros(len(MARGIN_CLASSES))
    cav_pos = cav_neg = 0
    can_pos = can_neg = 0
    for batch in loader:
        d = batch.density.numpy(); m = batch.margin.numpy()
        c = batch.cavit.numpy();   k = batch.cancer.numpy()
        for i in range(len(d)):
            if d[i] >= 0: dens_cnt[int(d[i])] += 1
            if m[i] >= 0: marg_cnt[int(m[i])] += 1
            if not np.isnan(c[i]):
                if c[i] > 0.5: cav_pos += 1
                else: cav_neg += 1
            if not np.isnan(k[i]):
                if k[i] > 0.5: can_pos += 1
                else: can_neg += 1

    def inv_freq(cnt):
        cnt = np.asarray(cnt, dtype=np.float32)
        cnt = np.where(cnt > 0, cnt, 1.0)
        w = cnt.sum() / (len(cnt) * cnt)
        return torch.tensor(w, dtype=torch.float32, device=device)

    return {
        "density": inv_freq(dens_cnt),
        "margin":  inv_freq(marg_cnt),
        "cavit":   torch.tensor([max(1.0, cav_neg / max(cav_pos, 1))],
                                  dtype=torch.float32, device=device),
        "cancer":  torch.tensor([max(1.0, can_neg / max(can_pos, 1))],
                                  dtype=torch.float32, device=device),
    }


@torch.no_grad()
def evaluate(model, loader, device,
             n_worst: int = 6, n_best: int = 6):
    """Return (metrics, panels, per_class_rows, per_sample_rows).

    metrics: scalar dict for log.jsonl
    panels: dict best/worst → list of sample dicts for nodule_viz.render
    per_class_rows: wandb.Table rows (group, class, support, n_pred, P, R, F1)
    per_sample_rows: wandb.Table rows (case_id, lesion_id, n_correct, GT/Pred per head)
    """
    model.eval()
    P_dens, P_margin, P_cavit, P_cancer = [], [], [], []
    T_dens, T_margin, T_cavit, T_cancer = [], [], [], []
    sample_buf: List[dict] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.image)
        pd = out["density"].argmax(-1).cpu().numpy()
        pm = out["margin"].argmax(-1).cpu().numpy()
        pc = (torch.sigmoid(out["cavit"]) > 0.5).cpu().numpy()
        pk = torch.sigmoid(out["cancer"]).cpu().numpy()
        td = batch.density.cpu().numpy()
        tm = batch.margin.cpu().numpy()
        tc = batch.cavit.cpu().numpy()
        tk = batch.cancer.cpu().numpy()
        P_dens.append(pd); P_margin.append(pm); P_cavit.append(pc); P_cancer.append(pk)
        T_dens.append(td); T_margin.append(tm); T_cavit.append(tc); T_cancer.append(tk)

        imgs = batch.image.cpu().numpy()
        for i in range(imgs.shape[0]):
            n_ok = 0
            n_ok += int(td[i] >= 0 and pd[i] == td[i])
            n_ok += int(tm[i] >= 0 and pm[i] == tm[i])
            n_ok += int(not np.isnan(tc[i]) and pc[i] == bool(tc[i]))
            n_ok += int((pk[i] > 0.5) == bool(tk[i]))
            sample_buf.append({
                "img": imgs[i].copy(),
                "case_id": batch.case_id[i],
                "lesion_id": batch.lesion_id[i],
                "gt_density":   DENSITY_CLASSES[int(td[i])] if td[i] >= 0 else None,
                "pred_density": DENSITY_CLASSES[int(pd[i])],
                "gt_margin":    MARGIN_CLASSES[int(tm[i])] if tm[i] >= 0 else None,
                "pred_margin":  MARGIN_CLASSES[int(pm[i])],
                "gt_cavit":     None if np.isnan(tc[i]) else int(tc[i]),
                "pred_cavit":   int(pc[i]),
                "gt_cancer":    int(tk[i]),
                "pred_cancer":  float(pk[i]),
                "n_correct":    n_ok,
            })
    P_dens = np.concatenate(P_dens); T_dens = np.concatenate(T_dens)
    P_margin = np.concatenate(P_margin); T_margin = np.concatenate(T_margin)
    P_cavit = np.concatenate(P_cavit); T_cavit = np.concatenate(T_cavit)
    P_cancer = np.concatenate(P_cancer); T_cancer = np.concatenate(T_cancer)

    def macro_f1(P, T, classes):
        keep = T >= 0
        if keep.sum() == 0:
            return 0.0
        P, T = P[keep], T[keep]
        f1s = []
        for k in range(len(classes)):
            tp = ((P == k) & (T == k)).sum()
            fp = ((P == k) & (T != k)).sum()
            fn = ((P != k) & (T == k)).sum()
            denom = 2 * tp + fp + fn
            f1s.append(0.0 if denom == 0 else float(2 * tp / denom))
        return float(np.mean(f1s))

    def acc(P, T):
        keep = T >= 0
        return float((P[keep] == T[keep]).mean()) if keep.sum() else 0.0

    def cavit_metrics(P, T):
        keep = ~np.isnan(T)
        if keep.sum() == 0:
            return 0.0, 0.0
        Pk, Tk = P[keep].astype(np.float32), T[keep].astype(np.float32)
        a = float((Pk == Tk).mean())
        tp = float((Pk * Tk).sum())
        fp = float((Pk * (1 - Tk)).sum())
        fn = float(((1 - Pk) * Tk).sum())
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else 2 * tp / denom
        return a, f1

    def auroc(P_prob, T):
        keep = ~np.isnan(T)
        if keep.sum() < 2 or len(np.unique(T[keep])) < 2:
            return float("nan")
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(T[keep], P_prob[keep]))

    cav_a, cav_f1 = cavit_metrics(P_cavit, T_cavit)
    metrics = {
        "val/acc_density": acc(P_dens, T_dens),
        "val/f1_density":  macro_f1(P_dens, T_dens, DENSITY_CLASSES),
        "val/acc_margin":  acc(P_margin, T_margin),
        "val/f1_margin":   macro_f1(P_margin, T_margin, MARGIN_CLASSES),
        "val/acc_cavit":   cav_a,
        "val/f1_cavit":    cav_f1,
        "val/auc_cancer":  auroc(P_cancer, T_cancer),
    }

    # Per-class support + P/R/F1 table -----------------------------------
    per_class_rows: List[list] = []
    def _add_mc(group_name, classes, P, T):
        keep = T >= 0
        if keep.sum() == 0:
            return
        Pk, Tk = P[keep], T[keep]
        for k, cn in enumerate(classes):
            tp = float(((Pk == k) & (Tk == k)).sum())
            fp = float(((Pk == k) & (Tk != k)).sum())
            fn = float(((Pk != k) & (Tk == k)).sum())
            support = float((Tk == k).sum())
            n_pred = float((Pk == k).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class_rows.append([group_name, cn,
                                    int(support), int(n_pred),
                                    round(prec, 3), round(rec, 3), round(f1, 3)])

    _add_mc("density", DENSITY_CLASSES, P_dens, T_dens)
    _add_mc("margin",  MARGIN_CLASSES, P_margin, T_margin)

    keep_c = ~np.isnan(T_cavit)
    if keep_c.sum():
        Pc = P_cavit[keep_c].astype(np.int32)
        Tc = T_cavit[keep_c].astype(np.int32)
        for k, cn in enumerate(["absent", "present"]):
            tp = float(((Pc == k) & (Tc == k)).sum())
            fp = float(((Pc == k) & (Tc != k)).sum())
            fn = float(((Pc != k) & (Tc == k)).sum())
            support = float((Tc == k).sum())
            n_pred = float((Pc == k).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class_rows.append(["cavitation", cn,
                                    int(support), int(n_pred),
                                    round(prec, 3), round(rec, 3), round(f1, 3)])

    # Cancer binary using 0.5 threshold.
    keep_k = ~np.isnan(T_cancer)
    if keep_k.sum():
        Pk_bin = (P_cancer[keep_k] > 0.5).astype(np.int32)
        Tk = T_cancer[keep_k].astype(np.int32)
        for k, cn in enumerate(["benign", "cancer"]):
            tp = float(((Pk_bin == k) & (Tk == k)).sum())
            fp = float(((Pk_bin == k) & (Tk != k)).sum())
            fn = float(((Pk_bin != k) & (Tk == k)).sum())
            support = float((Tk == k).sum())
            n_pred = float((Pk_bin == k).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class_rows.append(["cancer", cn,
                                    int(support), int(n_pred),
                                    round(prec, 3), round(rec, 3), round(f1, 3)])

    # Best / worst panels --------------------------------------------------
    sample_buf.sort(key=lambda s: -s["n_correct"])
    best = sample_buf[:n_best]
    worst = sample_buf[-n_worst:][::-1]
    panels = {"best": best, "worst": worst}

    per_sample_rows = [
        [s["case_id"], s["lesion_id"], s["n_correct"],
         s["gt_density"], s["pred_density"],
         s["gt_margin"],  s["pred_margin"],
         "—" if s["gt_cavit"] is None else ("present" if s["gt_cavit"] else "absent"),
         "present" if s["pred_cavit"] else "absent",
         "cancer" if s["gt_cancer"] else "benign",
         f"{s['pred_cancer']:.2f}"]
        for s in (best + worst)
    ]

    return metrics, panels, per_class_rows, per_sample_rows


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit-cases", type=int, default=None)
    ap.add_argument("overrides", nargs="*",
                    help="OmegaConf dot-path overrides, e.g. train.seed=2027")
    args = ap.parse_args()

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
        dev_str = "cpu"
    device = torch.device(dev_str)

    train_dl, val_dl = build_loaders(cfg, dataset_root,
                                     limit_cases=args.limit_cases)

    model = NoduleClassifier(cfg.model.params).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.model.name}  params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(),
                             lr=float(cfg.train.optimizer.lr),
                             weight_decay=float(cfg.train.optimizer.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg.train.epochs))
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    w = {k: float(cfg.loss[k]) for k in
         ("density_weight", "margin_weight", "cavit_weight", "cancer_weight")}

    use_cw = bool(cfg.loss.get("class_balanced", True))
    class_w: Dict[str, torch.Tensor] = {}
    if use_cw:
        class_w = compute_class_weights(train_dl, device)
        print(f"[class_weights] density={class_w['density'].cpu().numpy().round(2)}")
        print(f"[class_weights] margin ={class_w['margin'].cpu().numpy().round(2)}")
        print(f"[class_weights] cavit  ={class_w['cavit'].cpu().numpy().round(2)}")
        print(f"[class_weights] cancer ={class_w['cancer'].cpu().numpy().round(2)}")

    log_path = out_dir / "log.jsonl"
    log_f = open(log_path, "w")
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    wb = None
    log_cfg = OmegaConf.to_container(cfg.train.log, resolve=True) \
        if "log" in cfg.train else {}
    if log_cfg.get("backend") == "wandb":
        try:
            import wandb
            wb = wandb.init(
                project=log_cfg.get("wandb", {}).get("project", "lung-bmct"),
                name=cfg.name, dir=str(out_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=log_cfg.get("mode", "online"),
            )
            print(f"[wandb] run={wb.name}  url={wb.url}")
        except Exception as e:
            print(f"[wandb] disabled: {e!r}")
            wb = None

    best_score = -1.0
    for ep in range(int(cfg.train.epochs)):
        model.train()
        t0 = time.time()
        running = {k: 0.0 for k in ("loss", "density", "margin", "cavit", "cancer")}
        n = 0
        for batch in train_dl:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(batch.image)
                loss, parts = compute_loss(out, batch, w, class_w)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            float(cfg.train.grad_clip_norm))
            scaler.step(opt); scaler.update()
            bs = batch.image.size(0)
            running["loss"] += loss.item() * bs
            for k, v in parts.items():
                running[k] += v * bs
            n += bs
        sched.step()
        ep_train = {
            "epoch": ep,
            "train/loss": running["loss"] / max(n, 1),
            "train/density": running["density"] / max(n, 1),
            "train/margin":  running["margin"]  / max(n, 1),
            "train/cavit":   running["cavit"]   / max(n, 1),
            "train/cancer":  running["cancer"]  / max(n, 1),
            "lr": opt.param_groups[0]["lr"],
            "time_sec": time.time() - t0,
        }
        val, panels, per_class_rows, per_sample_rows = evaluate(
            model, val_dl, device)
        row = {**ep_train, **val}
        log_f.write(json.dumps(row) + "\n"); log_f.flush()

        # Viz + per-class table.
        try:
            from bmct.visualize.nodule_viz import log_panels as _log_n
            _log_n(wb, viz_dir, ep, panels, per_sample_rows)
        except Exception as e:
            print(f"  [viz] skip: {e!r}")
        if wb is not None and per_class_rows:
            import wandb as _wandb
            tbl = _wandb.Table(
                columns=["group", "class", "support", "n_pred",
                          "precision", "recall", "f1"],
                data=per_class_rows,
            )
            wb.log({"val/per_class_table": tbl}, step=ep)
        print(f"ep {ep+1:02d}/{int(cfg.train.epochs)}  "
              f"loss={row['train/loss']:.4f}  "
              f"AUC_cancer={row['val/auc_cancer']:.3f}  "
              f"f1_dens={row['val/f1_density']:.3f}  "
              f"f1_margin={row['val/f1_margin']:.3f}  "
              f"f1_cavit={row['val/f1_cavit']:.3f}  "
              f"{row['time_sec']:.1f}s")

        if wb is not None:
            wb.log({k: v for k, v in row.items()
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))},
                   step=ep)

        # Composite score for "best": cancer AUC weighted highest.
        score = row["val/auc_cancer"] if not np.isnan(row["val/auc_cancer"]) else 0.0
        score += 0.5 * (row["val/f1_density"] + row["val/f1_margin"])
        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(),
                        "epoch": ep, "metrics": val},
                       out_dir / "best.pt")

    log_f.close()
    if wb is not None:
        wb.finish()
    print(f"\n[done] log={log_path}  best_score={best_score:.4f}")


if __name__ == "__main__":
    main()
