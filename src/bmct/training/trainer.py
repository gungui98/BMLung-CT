"""Generic PyTorch trainer for the BMCT models.

Supports mixed precision, gradient clipping, cosine LR with warmup, early
stopping on a validation metric, and per-epoch logging of every loss term.
Checkpoints are written to ``output_dir/ckpt_last.pt`` (rolling) and
``output_dir/ckpt_best.pt`` (best validation AUC).

Consumes the ``train:`` block of ``configs/experiment/*.yaml`` (composed
from ``configs/train/default.yaml``), with sub-blocks ``optimizer``,
``scheduler``, ``early_stopping`` and scalar keys ``epochs``, ``batch_size``,
``grad_clip_norm``, ``amp``.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from bmct.data.dataset import MultitaskBatch
from bmct.evaluation.metrics import roc_auc
from bmct.losses import MaskedMultitaskLoss
from bmct.training.optim import build_optimizer, build_scheduler
from bmct.utils.logging import get_logger
from bmct.utils.wandb_logger import WandbLogger


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: MaskedMultitaskLoss,
        cfg: DictConfig,
        output_dir: str,
        device: str = "cuda",
        run_name: str = "bmct_run",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(str(self.output_dir / "train.log"))
        self.amp = bool(cfg.get("amp", True)) and torch.cuda.is_available()
        try:
            log_cfg = cfg.get("log", {})
        except Exception:
            log_cfg = {}
        self.wandb = WandbLogger(log_cfg, run_name=run_name, config=dict(cfg))

        _params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = build_optimizer(_params, cfg.optimizer) if _params else None
        self.scheduler = None  # built on fit() once we know steps/epoch
        self.scaler = GradScaler(enabled=self.amp)
        self.best_val = -math.inf
        es = cfg.get("early_stopping", {})
        self.patience = int(es.get("patience", 20)) if es else 20
        self.bad_epochs = 0

    # ---------------------------------------------------------------------
    def _step(self, batch: MultitaskBatch):
        batch = batch.to(self.device)
        with autocast(enabled=self.amp):
            outputs = self.model(batch.image, batch.clinical)
            logs = self.loss_fn(outputs, batch)
        return outputs, logs, batch

    # ---------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        running = defaultdict(float)
        n = 0
        clip = float(self.cfg.get("grad_clip_norm", 0.0))
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            _, logs, batch = self._step(batch)
            loss = logs["_total"]
            self.scaler.scale(loss).backward()
            if clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            bs = batch.image.size(0)
            for k, v in logs.items():
                if k.startswith("_"):
                    continue
                running[k] += float(v.detach().cpu()) * bs
            n += bs
        return {k: v / max(n, 1) for k, v in running.items()}

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader,
                    collect_vis: bool = False) -> Dict[str, float]:
        self.model.eval()
        logits, labels = [], []
        case_ids: list[str] = []
        ct_slices: list[torch.Tensor] = []
        seg_gts:   list[torch.Tensor] = []
        seg_prs:   list[torch.Tensor] = []
        running = defaultdict(float)
        n = 0
        for batch in loader:
            outputs, logs, batch = self._step(batch)
            logits.append(outputs["case_logit"].detach().cpu())
            labels.append(batch.case_label.detach().cpu())
            if collect_vis:
                # Median across K key-slices (lung-window channel) — robust to
                # the per-pixel saturated outliers that wash out mean/MIP views.
                ct_slices.append(batch.image[:, :, 0].median(dim=1).values.detach().cpu())
                seg_gts.append(batch.seg_target.squeeze(1).detach().cpu())
                if "seg_logits" in outputs:
                    sp = torch.sigmoid(outputs["seg_logits"]).detach().cpu()
                    if sp.dim() == 4:
                        sp = sp.squeeze(1)
                    seg_prs.append(sp)
                case_ids.extend(batch.case_id)
            bs = batch.image.size(0)
            for k, v in logs.items():
                if k.startswith("_"):
                    continue
                running[k] += float(v.detach().cpu()) * bs
            n += bs
        l = torch.cat(logits).numpy()
        y = torch.cat(labels).numpy()
        metrics = {k: v / max(n, 1) for k, v in running.items()}
        metrics["auc"] = roc_auc(y, l)
        if collect_vis:
            import numpy as np
            metrics["_vis"] = {
                "case_ids": case_ids,
                "y": y,
                "p": 1.0 / (1.0 + np.exp(-l)),
                "ct": torch.cat(ct_slices).numpy() if ct_slices else None,
                "seg_gt": torch.cat(seg_gts).numpy() if seg_gts else None,
                "seg_pr": torch.cat(seg_prs).numpy() if seg_prs else None,
            }
        return metrics

    # ---------------------------------------------------------------------
    def _fit_sklearn(self, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int,
                     X_train: "np.ndarray | None" = None,
                     y_train: "np.ndarray | None" = None,
                     X_val: "np.ndarray | None" = None,
                     y_val: "np.ndarray | None" = None) -> None:
        import numpy as np
        from bmct.evaluation.metrics import roc_auc
        if X_train is None:
            _X, _y = [], []
            for batch in train_loader:
                _X.append(batch.clinical.numpy())
                _y.append(batch.case_label.numpy())
            X_train = np.concatenate(_X)
            y_train = np.concatenate(_y)
        self.model.fit(X_train, y_train)
        if X_val is None:
            _X, _y = [], []
            for batch in val_loader:
                _X.append(batch.clinical.numpy())
                _y.append(batch.case_label.numpy())
            X_val = np.concatenate(_X)
            y_val = np.concatenate(_y)
        proba = self.model.predict_proba(X_val)
        val_auc = float(roc_auc(y_val, proba))
        self.logger.info(f"sklearn baseline | val_auc {val_auc:.4f}")
        torch.save({"sklearn_model": self.model._model, "val_auc": val_auc},
                   self.output_dir / "ckpt_best.pt")
        with open(self.output_dir / "metrics.jsonl", "w") as f:
            f.write(json.dumps({"epoch": 1, "val": {"auc": val_auc}}) + "\n")

    def load_checkpoint(self, path: str) -> int:
        """Restore model+optimizer+scheduler+epoch from a checkpoint.

        Returns the epoch number to resume from (0 if the checkpoint is a
        legacy state-dict-only file).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        if self.optimizer is not None and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scaler is not None and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.best_val = float(ckpt.get("best_val", ckpt.get("val_auc", -math.inf)))
        self.bad_epochs = int(ckpt.get("bad_epochs", 0))
        start_ep = int(ckpt.get("epoch", 0))
        self._pending_scheduler_state = ckpt.get("scheduler")
        self.logger.info(
            f"resumed from {path} @ epoch {start_ep}, best_val {self.best_val:.4f}"
        )
        return start_ep

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, start_epoch: int = 0) -> None:
        from bmct.models.baselines import _SklearnBaseline
        if isinstance(self.model, _SklearnBaseline):
            self._fit_sklearn(train_loader, val_loader, epochs)
            return
        self.scheduler = build_scheduler(
            self.optimizer, self.cfg.scheduler,
            total_epochs=epochs,
            steps_per_epoch=max(len(train_loader), 1),
        )
        pending = getattr(self, "_pending_scheduler_state", None)
        if pending is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(pending)
            except Exception as e:
                self.logger.warning(f"could not restore scheduler state: {e}")
            self._pending_scheduler_state = None
        for ep in range(start_epoch + 1, epochs + 1):
            train_logs = self._train_epoch(train_loader)
            collect_vis = self.wandb.enabled and self.wandb.log_best_worst_k > 0
            val_logs = self._eval_epoch(val_loader, collect_vis=collect_vis)
            vis = val_logs.pop("_vis", None)
            msg = (f"epoch {ep:03d} | "
                   f"train_loss {train_logs.get('loss', 0):.4f} | "
                   f"val_loss {val_logs.get('loss', 0):.4f} | "
                   f"val_auc {val_logs['auc']:.4f}")
            self.logger.info(msg)
            with open(self.output_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"epoch": ep,
                                    "train": train_logs,
                                    "val": val_logs}) + "\n")
            # wandb scalars + best/worst visualisations
            self.wandb.log(
                {**{f"train/{k}": v for k, v in train_logs.items()},
                 **{f"val/{k}":   v for k, v in val_logs.items()},
                 "epoch": ep},
                step=ep,
            )
            if vis is not None:
                self.wandb.log_best_worst(
                    case_ids=vis["case_ids"], y=vis["y"], p=vis["p"],
                    ct_slices=vis["ct"], seg_targets=vis["seg_gt"],
                    seg_preds=vis["seg_pr"], tag="val", step=ep,
                )

            def _ckpt_payload():
                return {
                    "state_dict": self.model.state_dict(),
                    "optimizer":  self.optimizer.state_dict() if self.optimizer else None,
                    "scheduler":  self.scheduler.state_dict() if self.scheduler else None,
                    "scaler":     self.scaler.state_dict() if self.scaler else None,
                    "epoch":      ep,
                    "best_val":   self.best_val,
                    "bad_epochs": self.bad_epochs,
                    "val_auc":    val_logs["auc"],
                }
            if val_logs["auc"] > self.best_val:
                self.best_val = val_logs["auc"]
                self.bad_epochs = 0
                torch.save(_ckpt_payload(), self.output_dir / "ckpt_best.pt")
            else:
                self.bad_epochs += 1
            torch.save(_ckpt_payload(), self.output_dir / "ckpt_last.pt")
            if self.bad_epochs >= self.patience:
                self.logger.info(f"Early stopping at epoch {ep}.")
                break
        self.wandb.finish()
