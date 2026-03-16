"""
Training / evaluation loop
  - AdamW optimizer + cosine LR scheduler
  - per-epoch CSV & TXT logs (includes LR)
  - best.pt checkpoint
  - val / test CSV+PNG export
"""

import time, json, torch, pandas as pd
from tqdm import trange
from pathlib import Path
from typing import Optional, Tuple
from sklearn.metrics import (mean_absolute_error, recall_score,
                             f1_score, balanced_accuracy_score)
from plot_utils import (plot_and_save_classification_results,
                        plot_loss_curves_from_logfile)
from config import CFG


class Engine:
    # ------------------------------------------------------------
    def __init__(self,
                 model: torch.nn.Module,
                 loaders: Tuple[torch.utils.data.DataLoader,
                                Optional[torch.utils.data.DataLoader],
                                torch.utils.data.DataLoader],
                 cfg: CFG):

        self.cfg    = cfg
        self.device = cfg.device
        self.model  = model.to(self.device)

        self.tr_dl, self.va_dl, self.te_dl = loaders

        # AdamW — correct weight decay for pretrained fine-tuning
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.wd
        )

        # Cosine LR scheduler — smoothly decays to 0 over all epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=cfg.epochs,
            eta_min=1e-6
        )

        self.loss_fn = torch.nn.L1Loss()

        # ── logging paths ────────────────────────────────────────
        self.exp_dir = cfg.exp_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.csv_log = self.exp_dir / "train_log.csv"
        self.txt_log = self.exp_dir / "train_log.txt"
        self.history = []

        with open(self.txt_log, "w") as f:
            f.write(f"Experiment dir: {self.exp_dir}\n")
            ws = getattr(cfg, "window_size", 0)
            f.write(f"Backbone: {cfg.backbone}  img_size={cfg.img_size}  window_size={ws if ws else 'default'}\n")
            f.write(f"Aggregator: {cfg.aggregator}  n_frames={cfg.n_frames}\n")
            f.write(f"Optimizer: AdamW  lr={cfg.lr}  wd={cfg.wd}\n")
            f.write(f"Scheduler: CosineAnnealingLR  T_max={cfg.epochs}  eta_min=1e-6\n\n")

    # ------------------------------------------------------------
    def _run_epoch(self, loader, train: bool):
        """Returns: avg_loss, preds(T), targets(T), ids(list[str])"""
        self.model.train(train)
        if not train:
            torch.cuda.empty_cache()

        preds, targets, ids = [], [], []
        step, running = 0, 0.0
        self.opt.zero_grad()

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for bag, y, vid in loader:
                bag = bag.to(self.device)
                y   = y.to(self.device).view(1)

                y_hat = self.model(bag)
                loss  = self.loss_fn(y_hat, y) / self.cfg.grad_accum

                if train:
                    loss.backward()
                    step += 1
                    if step % self.cfg.grad_accum == 0:
                        self.opt.step()
                        self.opt.zero_grad()

                running += loss.item() * self.cfg.grad_accum
                preds.append(y_hat.detach().cpu())
                targets.append(y.detach().cpu())
                ids.append(str(vid))

        if train and step % self.cfg.grad_accum:
            self.opt.step()
            self.opt.zero_grad()

        preds   = torch.cat(preds).flatten()
        targets = torch.cat(targets).flatten()
        return running / len(loader), preds, targets, ids

    # ------------------------------------------------------------
    def fit(self):
        n_train = len(self.tr_dl.dataset)
        n_val   = len(self.va_dl.dataset) if self.va_dl is not None else 0
        n_test  = len(self.te_dl.dataset)

        msg = f"Dataset sizes: Train={n_train}, Val={n_val}, Test={n_test}"
        print(msg)
        open(self.txt_log, "a").write(msg + "\n")

        best_metric = float("inf")
        best_epoch  = -1
        fold_t0     = time.time()

        agg = self.cfg.aggregator
        fold_tag = self.exp_dir.name  # e.g. fold00
        for epoch in trange(1, self.cfg.epochs + 1, desc=f"{agg}|{fold_tag}", unit="ep"):
            t0      = time.time()
            current_lr = self.scheduler.get_last_lr()[0] if epoch > 1 else self.cfg.lr

            tr_loss, tr_pred, tr_tgt, _ = self._run_epoch(self.tr_dl, train=True)

            # Step scheduler after each epoch
            self.scheduler.step()

            if self.va_dl is not None:
                _, v_pred, v_tgt, _ = self._run_epoch(self.va_dl, train=False)
                val_mae    = mean_absolute_error(v_tgt.numpy(), v_pred.numpy())
                key_metric = val_mae
            else:
                val_mae    = None
                key_metric = tr_loss

            epoch_sec = time.time() - t0
            gpu_mem   = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()

            # ── log ────────────────────────────────────────────
            msg = (
                f"E{epoch:03d}  tr_loss={tr_loss:.4f}  "
                f"{'val_mae='+f'{val_mae:.4f}  ' if val_mae is not None else ''}"
                f"lr={current_lr:.2e}  time={epoch_sec:.1f}s  gpu={gpu_mem:.0f}MB"
            )
            print(msg)
            open(self.txt_log, "a").write(msg + "\n")

            self.history.append(dict(
                epoch      = epoch,
                train_loss = tr_loss,
                val_mae    = val_mae,
                lr         = current_lr,
                sec        = epoch_sec,
                gpu_mb     = gpu_mem,
            ))

            # ── checkpoint ─────────────────────────────────────
            if key_metric < best_metric:
                best_metric, best_epoch = key_metric, epoch
                torch.save(self.model.state_dict(), self.exp_dir / "best.pt")

        pd.DataFrame(self.history).to_csv(self.csv_log, index=False)

        fold_total_sec = time.time() - fold_t0
        fin = (
            f"Finished. Best metric {best_metric:.4f} at epoch {best_epoch}. "
            f"Total fold time: {fold_total_sec/3600:.2f}h ({fold_total_sec:.0f}s)"
        )
        print(fin)
        open(self.txt_log, "a").write(fin + "\n")

    # ------------------------------------------------------------
    def _eval_and_dump(self, loader, df_src: pd.DataFrame, phase: str) -> dict:
        """Predict, save CSV+PNG, return full metrics dict."""
        # Load test/val videos into RAM if still lazy (avoids repeated disk reads)
        if hasattr(loader.dataset, 'load_into_ram'):
            loader.dataset.load_into_ram()

        self.model.load_state_dict(
            torch.load(self.exp_dir / "best.pt", map_location=self.device)
        )

        _, preds, targets, ids = self._run_epoch(loader, train=False)
        preds, targets = preds.numpy(), targets.numpy()

        vid_col = getattr(self.cfg, "video_id_col", "video")
        ids     = list(map(str, ids))
        idxed   = df_src.set_index(vid_col)
        missing = [i for i in ids if i not in idxed.index]
        if missing:
            print(f"[WARN] {len(missing)} ID not found in df_src: {missing[:5]}...")

        meta = idxed.reindex(ids)

        csv_path = self.exp_dir / f"{phase}_predicted_true_values.csv"
        png_path = self.exp_dir / f"{phase}_classification_summary.png"

        out_df = pd.DataFrame({
            vid_col:                        ids,
            self.cfg.stratify_col:          meta[self.cfg.stratify_col].values,
            "target":                       targets,
            "prediction":                   preds,
        })
        out_df.to_csv(csv_path, index=False)

        plot_loss_curves_from_logfile(self.txt_log,
                                      save_path=self.txt_log.with_suffix(".png"))
        plot_and_save_classification_results(str(csv_path), str(png_path))

        bin_t = (targets > self.cfg.threshold).astype(int)
        bin_p = (preds   > self.cfg.threshold).astype(int)

        return dict(
            phase           = phase,
            MAE             = float(mean_absolute_error(targets, preds)),
            F1              = float(f1_score(bin_t, bin_p, zero_division=0)),
            Recall          = float(recall_score(bin_t, bin_p, zero_division=0)),
            BalancedAcc     = float(balanced_accuracy_score(bin_t, bin_p)),
        )

    # ------------------------------------------------------------
    def evaluate_test(self, tst_loader, tst_df):
        metrics = self._eval_and_dump(tst_loader, tst_df, "test")
        print(json.dumps(metrics, indent=2))
        return metrics

    # ------------------------------------------------------------
    def evaluate_val_test(self,
                          val_loader: Optional[torch.utils.data.DataLoader],
                          val_df:     Optional[pd.DataFrame],
                          tst_loader: torch.utils.data.DataLoader,
                          tst_df:     pd.DataFrame):
        out = {}
        if val_loader is not None and val_df is not None:
            out["val"]  = self._eval_and_dump(val_loader, val_df, "val")
        out["test"] = self._eval_and_dump(tst_loader, tst_df, "test")
        print(json.dumps(out, indent=2))
        return out

    # ------------------------------------------------------------
    def evaluate_all(self, val_df: Optional[pd.DataFrame], tst_df: pd.DataFrame):
        return self.evaluate_val_test(self.va_dl, val_df, self.te_dl, tst_df)