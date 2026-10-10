"""Stage-2 nodule classifier visualizations.

Per epoch: render N "best" (all heads correct) + N "worst" (most heads wrong)
nodule crops with image + Vietnamese GT/Pred labels overlaid as a translucent
text box. Logged as separate wandb panels ``val/nodule_best`` and
``val/nodule_worst``, plus a per-sample score table ``val/nodule_table``.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


CLASS_VI = {
    "density":  {"solid": "đặc", "part_solid": "bán-đặc",
                  "ground_glass": "kính mờ", "consolidation": "đông đặc"},
    "margin":   {"smooth": "bờ nhẵn", "lobulated": "múi",
                  "spiculated": "tua gai", "irregular": "không đều"},
    "cavitation": {"absent": "không hang", "present": "có hang"},
    "cancer":   {0: "lành", 1: "ác"},
}
GROUP_VI = {
    "density": "Đậm độ",
    "margin":  "Bờ",
    "cavit":   "Hang",
    "cancer":  "Ác/lành",
}


def _vi(group: str, raw) -> str:
    if raw is None:
        return "—"
    if group in ("density", "margin"):
        # raw is string class name
        m = CLASS_VI[group]
        return m.get(raw, str(raw))
    if group == "cavit":
        return CLASS_VI["cavitation"].get("present" if raw else "absent", "—")
    if group == "cancer":
        return CLASS_VI["cancer"].get(int(raw), "—")
    return str(raw)


def render_panel(sample: dict) -> np.ndarray:
    """Panel for one nodule: image + VI GT/Pred text overlay.

    sample keys: img (3,H,W), case_id, lesion_id, gt_density (str or None),
                 pred_density (str), gt_margin, pred_margin, gt_cavit (0/1
                 or None), pred_cavit (0/1), gt_cancer (0/1), pred_cancer
                 (float prob), n_correct (0–4).
    """
    import matplotlib.pyplot as plt

    img = sample["img"][1]                                # centre slice
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=100)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

    rows = []
    for g, gt_key, pr_key in [
        ("density", "gt_density", "pred_density"),
        ("margin",  "gt_margin",  "pred_margin"),
        ("cavit",   "gt_cavit",   "pred_cavit"),
        ("cancer",  "gt_cancer",  "pred_cancer"),
    ]:
        gt = sample.get(gt_key); pr = sample.get(pr_key)
        # Cancer pred is a float probability, others are class indices/strings.
        if g == "cancer" and isinstance(pr, (int, float, np.floating)):
            pr_str = f"{_vi('cancer', 1 if pr>0.5 else 0)} (p={float(pr):.2f})"
        else:
            pr_str = _vi(g, pr)
        gt_str = _vi(g, gt)
        ok = "✓" if (gt is not None and ((g == "cancer" and (pr is not None) and ((pr > 0.5) == bool(gt)))
                                          or (g != "cancer" and pr == gt))) else "✗"
        rows.append(f"{GROUP_VI[g]:<10} GT: {gt_str:<14}  Pred: {pr_str}  {ok}")

    txt = "\n".join(rows)
    ax.text(0.02, 0.02, txt, transform=ax.transAxes,
            va="bottom", ha="left", color="white",
            family="DejaVu Sans", fontsize=7,
            bbox=dict(facecolor="black", alpha=0.6,
                      edgecolor="none", boxstyle="round,pad=0.3"))
    title = f"{sample['case_id']} / {sample['lesion_id']}  ({sample['n_correct']}/4)"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def render_grid(samples: List[dict]) -> np.ndarray:
    if not samples:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    # Lay out in a 2-column grid for compactness.
    panels = [render_panel(s) for s in samples]
    pairs = []
    for i in range(0, len(panels), 2):
        if i + 1 < len(panels):
            row = np.concatenate([panels[i], panels[i + 1]], axis=1)
        else:
            row = panels[i]
        pairs.append(row)
    h = max(r.shape[0] for r in pairs)
    w = max(r.shape[1] for r in pairs)
    canvas = np.zeros((h * len(pairs), w, 3), dtype=np.uint8)
    for j, r in enumerate(pairs):
        canvas[j * h: j * h + r.shape[0], : r.shape[1]] = r
    return canvas


def log_panels(wb, viz_dir, ep: int, panels: Dict[str, List[dict]],
               table_rows: List[list]) -> None:
    """Save PNGs locally + log to wandb."""
    import imageio.v2 as imageio
    wb_images = {}
    for g_name, samples in panels.items():
        grid = render_grid(samples)
        png_path = viz_dir / f"epoch_{ep+1:03d}_{g_name}.png"
        imageio.imwrite(png_path, grid)
        if wb is not None:
            import wandb as _wandb
            wb_images[f"val/nodule_{g_name}"] = _wandb.Image(str(png_path))
    if wb is not None and wb_images:
        wb.log(wb_images, step=ep)
    if wb is not None and table_rows:
        import wandb as _wandb
        tbl = _wandb.Table(
            columns=["case_id", "lesion_id", "n_correct",
                      "GT_density", "Pred_density",
                      "GT_margin",  "Pred_margin",
                      "GT_cavit",   "Pred_cavit",
                      "GT_cancer",  "Pred_cancer"],
            data=table_rows,
        )
        wb.log({"val/nodule_table": tbl}, step=ep)
