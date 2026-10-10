"""Minimal Weights & Biases integration for BMCT runs.

Lazy: only imports wandb when ``backend == "wandb"``.  Used by both the trainer
(per-epoch val metrics + best/worst case visualization) and evaluate.py (final
eval visualization).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


class WandbLogger:
    """Thin wrapper around wandb that no-ops when disabled."""

    def __init__(self, log_cfg, run_name: str, config: dict) -> None:
        backend = str(log_cfg.get("backend", "csv")) if log_cfg else "csv"
        self.enabled = backend == "wandb"
        self.run = None
        self.log_best_worst_k = 0
        if not self.enabled:
            return
        try:
            import wandb  # noqa: F401
            wb_cfg = dict(log_cfg.get("wandb", {})) if log_cfg.get("wandb") else {}
            self.log_best_worst_k = int(wb_cfg.get("log_best_worst_k", 0))
            kwargs = dict(
                project=wb_cfg.get("project", "lung-bmct"),
                name=run_name,
                config=_jsonable(config),
                reinit=True,
            )
            entity = wb_cfg.get("entity")
            if entity:
                kwargs["entity"] = entity
            self.run = wandb.init(**kwargs)
        except Exception as e:
            print(f"[wandb] init failed ({e}); falling back to no-op logger")
            self.enabled = False
            self.run = None

    # ------------------------------------------------------------------
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        import wandb
        wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    def log_best_worst(
        self,
        *,
        case_ids: Sequence[str],
        y: np.ndarray,
        p: np.ndarray,
        ct_slices: Optional[np.ndarray] = None,         # (N, H, W) central slice
        seg_targets: Optional[np.ndarray] = None,        # (N, H, W) GT mask
        seg_preds: Optional[np.ndarray] = None,          # (N, H, W) predicted mask (sigmoid)
        tag: str = "val",
        step: Optional[int] = None,
    ) -> None:
        """Log top-K best (high-conf correct) and worst (high-conf wrong) cases.

        "best"  = correct & |p - 0.5| largest
        "worst" = wrong   & |p - 0.5| largest
        """
        if not self.enabled or self.log_best_worst_k <= 0:
            return
        import wandb
        k = self.log_best_worst_k
        pred = (p >= 0.5).astype(int)
        correct = pred == y
        conf = np.abs(p - 0.5)
        # rank
        best_idx = np.argsort(-conf * correct)[:k]                 # correct, high conf
        worst_idx = np.argsort(-conf * (~correct).astype(float))[:k]  # wrong, high conf
        best_idx = [i for i in best_idx if correct[i]]
        worst_idx = [i for i in worst_idx if not correct[i]]

        def _dice(gt, pr) -> float:
            if gt is None or pr is None:
                return float("nan")
            g = (gt > 0.5).astype(np.float32)
            r = (pr > 0.5).astype(np.float32)
            denom = g.sum() + r.sum()
            return float(2.0 * (g * r).sum() / denom) if denom > 0 else float("nan")

        if ct_slices is None:
            return

        def _tile(i, color):
            """Render one CT tile with overlays + burn-in text. color: 'best'|'worst'."""
            cid = case_ids[i]
            mask_gt = seg_targets[i] if seg_targets is not None else None
            mask_pr = seg_preds[i] if seg_preds is not None else None
            dice = _dice(mask_gt, mask_pr)
            img = _norm01(ct_slices[i])
            tile_rgb = (np.stack([img] * 3, axis=-1) * 255).astype(np.uint8)
            # blend GT (green) + Pred (red) overlays
            alpha = 0.45
            if mask_gt is not None:
                m = mask_gt > 0.5
                tile_rgb[m] = (1 - alpha) * tile_rgb[m] + alpha * np.array([40, 220, 40])
            if mask_pr is not None:
                m = mask_pr > 0.5
                tile_rgb[m] = (1 - alpha) * tile_rgb[m] + alpha * np.array([220, 40, 40])
            burn = f"{color} {cid}  p={p[i]:.2f}  dice={dice:.2f}"
            return _burn_text_rgb(tile_rgb, burn)

        all_tiles = [_tile(i, "best") for i in best_idx] + \
                    [_tile(i, "worst") for i in worst_idx]
        if not all_tiles:
            return
        composite = _grid(all_tiles, n_cols=min(4, len(all_tiles)))
        wandb.log({f"{tag}/best_worst": wandb.Image(composite)}, step=step)

    # ------------------------------------------------------------------
    def finish(self) -> None:
        if self.enabled and self.run is not None:
            import wandb
            wandb.finish()


def _jsonable(o):
    """Recursively coerce OmegaConf / dataclass / tensor into json-friendly types."""
    try:
        from omegaconf import OmegaConf, DictConfig, ListConfig
        if isinstance(o, (DictConfig, ListConfig)):
            return OmegaConf.to_container(o, resolve=True)
    except Exception:
        pass
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _burn_text(img01: np.ndarray, text: str) -> np.ndarray:
    """Convert a [0,1] HxW float image to uint8 RGB with text overlaid."""
    rgb = (np.stack([img01] * 3, axis=-1) * 255).astype(np.uint8)
    return _burn_text_rgb(rgb, text)


def _burn_text_rgb(rgb: np.ndarray, text: str) -> np.ndarray:
    """Burn text into an existing uint8 RGB image (top-left, on dark strip)."""
    from PIL import Image, ImageDraw
    pim = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pim)
    pad = max(4, pim.width // 100)
    draw.rectangle((0, 0, pim.width, pad * 4 + 8), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 0))
    return np.asarray(pim)


def _grid(tiles: list[np.ndarray], n_cols: int) -> np.ndarray:
    """Tile a list of equal-shape HxWx3 uint8 images into a grid."""
    n = len(tiles)
    n_rows = (n + n_cols - 1) // n_cols
    h, w = tiles[0].shape[:2]
    canvas = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, n_cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return canvas


def _norm01(a: np.ndarray) -> np.ndarray:
    """Per-image rendering normalization: histogram-equalize to [0,1].

    Robust to saturation from prior windowing — distributes whatever values
    are present across the displayable range. Good for inspection panels
    where we care about local contrast inside the lung parenchyma rather
    than absolute HU fidelity.
    """
    a = a.astype(np.float32)
    flat = a.flatten()
    if flat.max() - flat.min() < 1e-6:
        return np.zeros_like(a)
    # Sorted-rank histogram equalization.
    order = np.argsort(flat)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(flat), dtype=np.float32)
    return ranks.reshape(a.shape)
