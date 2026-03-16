# main.py
import argparse
import copy
import gc
import pandas as pd
import torch
from model import MILLabModel
from data import build_loaders
from engine import Engine
from config import CFG, set_seed


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregator",    required=True,
                   choices=["abmil", "transmil", "clamsb", "dsmil"])
    p.add_argument("--run_name",      default=None)
    p.add_argument("--n_folds",       type=int, default=CFG.n_folds)
    p.add_argument("--stratify_col",  default="stage")
    p.add_argument("--csv_path",      default=None)
    p.add_argument("--set_col",       default=None)
    p.add_argument("--fold_set_prefix", default=None)
    p.add_argument("--n_frames",      type=int, default=None)
    p.add_argument("--img_size",      type=int, default=None)
    p.add_argument("--embed_dim",     type=int, default=512)
    p.add_argument("--fold_idx",      type=int, default=-1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def train_one_fold(base_cfg: CFG, fold_idx: int | None):
    cfg = copy.deepcopy(base_cfg)

    set_seed(cfg.seed)
    print(f"[SEED] seed={cfg.seed} (fixed for all folds and aggregators)")

    if cfg.n_folds > 0 and fold_idx is not None:
        cfg.exp_dir = cfg.exp_dir / f"fold{fold_idx:02d}"
        cfg.exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Fold dir: {cfg.exp_dir}")

    tr_dl, val_dl, te_dl, tr_df, val_df, tst_df = build_loaders(cfg, fold_idx)

    model = MILLabModel(cfg)
    eng   = Engine(model, (tr_dl, val_dl, te_dl), cfg)

    try:
        eng.fit()
        metrics = eng.evaluate_val_test(val_dl, val_df, te_dl, tst_df)
        return metrics
    finally:
        del eng, model, tr_dl, val_dl, te_dl, tr_df, val_df, tst_df
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    base_cfg = CFG(
        aggregator   = args.aggregator,
        run_name     = args.run_name,
        n_folds      = args.n_folds,
        stratify_col = args.stratify_col,
        img_size     = args.img_size  if args.img_size  is not None else 224,
        n_frames     = args.n_frames  if args.n_frames  is not None else 600,
        embed_dim    = args.embed_dim,
    )

    if args.csv_path        is not None: base_cfg.csv_path         = args.csv_path
    if args.set_col         is not None: base_cfg.set_col          = args.set_col
    if args.fold_set_prefix is not None: base_cfg.fold_set_prefix  = args.fold_set_prefix

    if base_cfg.n_folds == 0:
        train_one_fold(base_cfg, fold_idx=None)
        return

    # Single fold mode (used by run_all.py)
    if args.fold_idx >= 0:
        print(f"\n===== Fold {args.fold_idx + 1}/{base_cfg.n_folds} (single) =====")
        train_one_fold(base_cfg, args.fold_idx)
        return

    # Sequential all-folds mode (fallback)
    fold_metrics = []
    for f in range(base_cfg.n_folds):
        print(f"\n===== Fold {f + 1}/{base_cfg.n_folds} =====")
        fold_metrics.append(train_one_fold(base_cfg, f))

    df_metrics = pd.DataFrame(fold_metrics)
    print("\nK-fold mean metrics:\n", df_metrics.mean(numeric_only=True))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
