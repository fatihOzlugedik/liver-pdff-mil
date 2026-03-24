"""
Training and evaluation engine.
"""

import time
import json
import torch
import torch.nn as nn
import pandas as pd
from tqdm import trange
from typing import Optional, Tuple
from sklearn.metrics import (
    mean_absolute_error, recall_score, f1_score,
    balanced_accuracy_score, accuracy_score
)
from plot_utils import plot_and_save_classification_results, plot_loss_curves_from_logfile
from config import CFG
from losses import build_loss, build_loss_from_preset


class Engine:
    def __init__(self,
                 model: nn.Module,
                 loaders: Tuple,
                 cfg: CFG):

        self.cfg = cfg
        self.device = cfg.device
        self.model = model.to(self.device)
        self.tr_dl, self.va_dl, self.te_dl = loaders

        # Multi-task config
        self.use_classifier = cfg.use_classifier
        self.cls_weight = cfg.cls_weight
        self.reg_weight = 1.0 - cfg.cls_weight

        # Optimizer and scheduler
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.epochs, eta_min=1e-6)

        # Loss functions
        self.reg_loss_fn = self._build_loss(cfg)
        self.cls_loss_fn = nn.CrossEntropyLoss()

        # Logging
        self.exp_dir = cfg.exp_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.csv_log = self.exp_dir / "train_log.csv"
        self.txt_log = self.exp_dir / "train_log.txt"
        self.history = []

        self._write_header(cfg)

    def _build_loss(self, cfg: CFG):
        if cfg.loss_preset:
            return build_loss_from_preset(cfg.loss_preset)
        return build_loss(cfg.loss_fn, **(cfg.loss_config or {}))

    def _write_header(self, cfg: CFG):
        with open(self.txt_log, "w") as f:
            f.write(f"Experiment: {self.exp_dir}\n")
            f.write(f"Aggregator: {cfg.aggregator}\n")
            f.write(f"Loss: {cfg.loss_preset or cfg.loss_fn}\n")
            if self.use_classifier:
                f.write(f"Classifier: weight={self.cls_weight:.2f}\n")
            f.write(f"LR: {cfg.lr}, WD: {cfg.wd}, Epochs: {cfg.epochs}\n\n")

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        if not train:
            torch.cuda.empty_cache()

        reg_preds, targets, cls_preds, cls_targets, ids = [], [], [], [], []
        running_loss = 0.0
        step = 0
        self.opt.zero_grad()

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for bag, y, cls_y, vid in loader:
                bag = bag.to(self.device)
                y = y.to(self.device).view(1)
                cls_y = cls_y.to(self.device).view(1)

                output = self.model(bag)

                if self.use_classifier:
                    y_hat, cls_logits = output
                    reg_loss = self.reg_loss_fn(y_hat, y)
                    cls_loss = self.cls_loss_fn(cls_logits.unsqueeze(0), cls_y)
                    loss = (self.reg_weight * reg_loss + self.cls_weight * cls_loss) / self.cfg.grad_accum
                    cls_preds.append(cls_logits.argmax().detach().cpu())
                    cls_targets.append(cls_y.detach().cpu())
                else:
                    y_hat = output
                    loss = self.reg_loss_fn(y_hat, y) / self.cfg.grad_accum

                if train:
                    loss.backward()
                    step += 1
                    if step % self.cfg.grad_accum == 0:
                        self.opt.step()
                        self.opt.zero_grad()

                running_loss += loss.item() * self.cfg.grad_accum
                reg_preds.append(y_hat.detach().cpu())
                targets.append(y.detach().cpu())
                ids.append(str(vid))

        if train and step % self.cfg.grad_accum:
            self.opt.step()
            self.opt.zero_grad()

        reg_preds = torch.cat(reg_preds).flatten()
        targets = torch.cat(targets).flatten()
        cls_preds = torch.stack(cls_preds).flatten() if cls_preds else None
        cls_targets = torch.cat(cls_targets).flatten() if cls_targets else None

        return running_loss / len(loader), reg_preds, targets, cls_preds, cls_targets, ids

    def fit(self):
        print(f"Train: {len(self.tr_dl.dataset)}, Val: {len(self.va_dl.dataset) if self.va_dl else 0}, Test: {len(self.te_dl.dataset)}")

        best_metric = float("inf")
        best_epoch = -1

        for epoch in trange(1, self.cfg.epochs + 1, desc=self.cfg.aggregator):
            t0 = time.time()
            lr = self.scheduler.get_last_lr()[0]

            tr_loss, _, _, _, _, _ = self._run_epoch(self.tr_dl, train=True)
            self.scheduler.step()

            val_mae, val_cls_acc = None, None
            if self.va_dl:
                _, v_pred, v_tgt, v_cls_pred, v_cls_tgt, _ = self._run_epoch(self.va_dl, train=False)
                val_mae = mean_absolute_error(v_tgt.numpy(), v_pred.numpy())
                if self.use_classifier and v_cls_pred is not None:
                    val_cls_acc = accuracy_score(v_cls_tgt.numpy(), v_cls_pred.numpy())

            key_metric = val_mae if val_mae else tr_loss
            if key_metric < best_metric:
                best_metric, best_epoch = key_metric, epoch
                torch.save(self.model.state_dict(), self.exp_dir / "best.pt")

            # Log
            msg = f"E{epoch:03d} loss={tr_loss:.4f}"
            if val_mae: msg += f" mae={val_mae:.4f}"
            if val_cls_acc: msg += f" cls={val_cls_acc:.3f}"
            msg += f" lr={lr:.2e} t={time.time()-t0:.1f}s"
            print(msg)
            open(self.txt_log, "a").write(msg + "\n")

            self.history.append({"epoch": epoch, "train_loss": tr_loss, "val_mae": val_mae, "val_cls_acc": val_cls_acc})

        pd.DataFrame(self.history).to_csv(self.csv_log, index=False)
        print(f"Best: {best_metric:.4f} @ epoch {best_epoch}")

    def _eval_and_dump(self, loader, df_src: pd.DataFrame, phase: str) -> dict:
        if hasattr(loader.dataset, 'load_into_ram'):
            loader.dataset.load_into_ram()

        self.model.load_state_dict(torch.load(self.exp_dir / "best.pt", map_location=self.device))
        _, preds, targets, cls_preds, cls_targets, ids = self._run_epoch(loader, train=False)
        preds, targets = preds.numpy(), targets.numpy()

        # Save predictions
        out_df = pd.DataFrame({
            self.cfg.video_id_col: ids,
            "target": targets,
            "prediction": preds,
        })
        if self.use_classifier and cls_preds is not None:
            out_df["cls_target"] = cls_targets.numpy()
            out_df["cls_pred"] = cls_preds.numpy()

        csv_path = self.exp_dir / f"{phase}_predictions.csv"
        out_df.to_csv(csv_path, index=False)

        # Plots
        try:
            plot_loss_curves_from_logfile(self.txt_log, save_path=self.txt_log.with_suffix(".png"))
            plot_and_save_classification_results(str(csv_path), str(self.exp_dir / f"{phase}_summary.png"))
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")

        # Metrics
        bin_t = (targets > self.cfg.threshold).astype(int)
        bin_p = (preds > self.cfg.threshold).astype(int)

        metrics = {
            "phase": phase,
            "MAE": float(mean_absolute_error(targets, preds)),
            "F1": float(f1_score(bin_t, bin_p, zero_division=0)),
            "Recall": float(recall_score(bin_t, bin_p, zero_division=0)),
            "BalancedAcc": float(balanced_accuracy_score(bin_t, bin_p)),
        }

        if self.use_classifier and cls_preds is not None:
            metrics["Cls_Acc"] = float(accuracy_score(cls_targets.numpy(), cls_preds.numpy()))
            metrics["Cls_F1"] = float(f1_score(cls_targets.numpy(), cls_preds.numpy(), average='weighted', zero_division=0))

        return metrics

    def evaluate_val_test(self, val_loader, val_df, tst_loader, tst_df):
        out = {}
        if val_loader:
            out["val"] = self._eval_and_dump(val_loader, val_df, "val")
        out["test"] = self._eval_and_dump(tst_loader, tst_df, "test")
        print(json.dumps(out, indent=2))
        return out
