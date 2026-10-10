"""Shared per-epoch visualization for slice-level training.

Both ``train_slice.py`` (seg-only) and ``train_slice_mt.py`` (multi-task)
use this module so the wandb panel layout stays consistent — 3 separate
panes (worst_pos / best_pos / worst_neg), each a vertical grid of samples
with 2 columns: error map (TP/FP/FN) and sigmoid probability heatmap.

Multi-task callers pass ``attr_overlay=True`` and the panel will add an
extra Vietnamese GT/Pred attr text block on top-left of the error map.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from bmct.data.schema import PROPERTY_GROUPS
from bmct.data.slice_dataset import GROUP_OFFSETS

GROUP_VI = {
    "density":              "Đậm độ",
    "margin":               "Bờ",
    "lobe":                 "Vị trí",
    "inflammatory_pattern": "Pattern viêm",
    "cavitation":           "Hang",
}
CLASS_VI = {
    "density": {
        "solid":         "đặc",
        "part_solid":    "bán-đặc",
        "ground_glass":  "kính mờ",
        "consolidation": "đông đặc",
    },
    "margin": {
        "smooth":     "bờ nhẵn",
        "lobulated":  "múi",
        "spiculated": "tua gai",
        "irregular":  "không đều",
    },
    "lobe": {
        "RUL": "thùy trên P", "RML": "thùy giữa P", "RLL": "thùy dưới P",
        "LUL": "thùy trên T", "LLL": "thùy dưới T",
    },
    "inflammatory_pattern": {
        "none":            "không",
        "peripheral_halo": "quầng viêm",
        "consolidation":   "đông đặc",
        "tree_in_bud":     "tree-in-bud",
    },
    "cavitation": {"absent": "không hang", "present": "có hang"},
}


def _attr_lines(vec: np.ndarray, thr: float = 0.5) -> List[str]:
    lines = []
    for g, classes in PROPERTY_GROUPS.items():
        s, e = GROUP_OFFSETS[g]
        sub = vec[s:e]
        sel = [CLASS_VI[g].get(classes[i], classes[i])
               for i, p in enumerate(sub) if p > thr]
        rhs = " · ".join(sel) if sel else "—"
        lines.append(f"{GROUP_VI[g]}: {rhs}")
    return lines


def render_panel(sample: dict, group: str,
                 attr_overlay: bool = False) -> np.ndarray:
    """2-up figure: error map (TP/FP/FN, optional attr overlay) | heatmap."""
    import matplotlib.pyplot as plt

    img = sample["img"]; gt = sample["gt"]; prob = sample["prob"]
    pred = prob > 0.5
    gt_b = gt > 0.5
    tp = pred & gt_b
    fp = pred & ~gt_b
    fn = ~pred & gt_b

    err = np.stack([img, img, img], axis=-1)
    err[tp] = 0.45 * err[tp] + 0.55 * np.array([0.0, 1.0, 0.0])
    err[fp] = 0.45 * err[fp] + 0.55 * np.array([1.0, 0.0, 0.0])
    err[fn] = 0.45 * err[fn] + 0.55 * np.array([1.0, 1.0, 0.0])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.0), dpi=100)
    axes[0].imshow(err)
    axes[0].set_title(f"err TP={int(tp.sum())} FP={int(fp.sum())} "
                       f"FN={int(fn.sum())}", fontsize=9)
    axes[1].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(prob, cmap="jet", vmin=0, vmax=1, alpha=0.5)
    axes[1].set_title("sigmoid heatmap", fontsize=9)
    for ax in axes:
        ax.axis("off")

    if attr_overlay and "attr_gt" in sample and "attr_prob" in sample:
        gt_lines = _attr_lines(sample["attr_gt"])
        pr_lines = _attr_lines(sample["attr_prob"])
        txt = "GT\n" + "\n".join(gt_lines) + "\n\nPred\n" + "\n".join(pr_lines)
        axes[0].text(
            0.02, 0.98, txt, transform=axes[0].transAxes,
            va="top", ha="left", color="white",
            family="DejaVu Sans", fontsize=6.5,
            bbox=dict(facecolor="black", alpha=0.55,
                      edgecolor="none", boxstyle="round,pad=0.3"),
        )

    title = (f"[{group}] {sample['case_id']} z={sample['slice_idx']}  "
             f"dice={sample['dice']:.3f}  fp_px={sample['fp_px']}")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def render_group_grid(samples: List[dict], group: str,
                      attr_overlay: bool = False) -> np.ndarray:
    if not samples:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return np.concatenate(
        [render_panel(s, group, attr_overlay) for s in samples], axis=0
    )


def log_panels(wb, viz_dir, ep: int, panels: Dict[str, List[dict]],
               attr_overlay: bool = False) -> None:
    """Save 3 PNGs locally + log each as a separate wandb image key,
    plus a per-sample score table aggregating worst_pos / best_pos / worst_neg.

    ``panels`` maps group_name → list of sample dicts (already top-K filtered).
    """
    import imageio.v2 as imageio
    wb_images = {}
    rows = []
    for g_name, samples in panels.items():
        grid = render_group_grid(samples, g_name, attr_overlay)
        png_path = viz_dir / f"epoch_{ep+1:03d}_{g_name}.png"
        imageio.imwrite(png_path, grid)
        if wb is not None:
            import wandb as _wandb
            wb_images[f"val/{g_name}"] = _wandb.Image(str(png_path))
        for s in samples:
            rows.append([
                g_name,
                s["case_id"], int(s["slice_idx"]),
                round(float(s["dice"]), 4),
                int(s["fp_px"]),
                int((s["gt"] > 0.5).sum()),
                int((s["prob"] > 0.5).sum()),
            ])
    if wb is not None and wb_images:
        wb.log(wb_images, step=ep)
    if wb is not None and rows:
        import wandb as _wandb
        tbl = _wandb.Table(
            columns=["group", "case_id", "slice", "dice", "fp_px",
                      "gt_px", "pred_px"],
            data=rows,
        )
        wb.log({"val/score_table": tbl}, step=ep)
