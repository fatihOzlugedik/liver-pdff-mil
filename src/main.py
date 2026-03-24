#!/usr/bin/env python3
"""
Train MIL model for PDFF regression.

Usage:
    python main.py configs/experiment.yaml
"""

import sys
import copy
import gc
import torch
import pandas as pd
from pathlib import Path
from model import MILModel
from data import build_loaders
from engine import Engine
from config import load_config, set_seed


def train_one_fold(base_cfg, fold_idx: int | None):
    """Train a single fold."""
    cfg = copy.deepcopy(base_cfg)
    set_seed(cfg.seed)

    if cfg.n_folds > 0 and fold_idx is not None:
        cfg.exp_dir = cfg.exp_dir / f"fold{fold_idx:02d}"
        cfg.exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[FOLD {fold_idx + 1}/{cfg.n_folds}]")

    tr_dl, val_dl, te_dl, tr_df, val_df, tst_df = build_loaders(cfg, fold_idx)
    model = MILModel(cfg)
    engine = Engine(model, (tr_dl, val_dl, te_dl), cfg)

    try:
        engine.fit()
        return engine.evaluate_val_test(val_dl, val_df, te_dl, tst_df)
    finally:
        del engine, model, tr_dl, val_dl, te_dl
        torch.cuda.empty_cache()
        gc.collect()


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config.yaml>")
        print("\nExample configs in configs/ directory")
        sys.exit(1)

    yaml_path = sys.argv[1]
    if not Path(yaml_path).exists():
        print(f"Config not found: {yaml_path}")
        sys.exit(1)

    cfg = load_config(yaml_path)

    print("\n" + "=" * 50)
    print(f"Aggregator:  {cfg.aggregator}")
    print(f"Loss:        {cfg.loss_preset or cfg.loss_fn}")
    print(f"Classifier:  {cfg.use_classifier} (w={cfg.cls_weight})")
    print(f"Epochs:      {cfg.epochs}")
    print(f"Folds:       {cfg.n_folds}")
    print("=" * 50 + "\n")

    if cfg.n_folds == 0:
        train_one_fold(cfg, None)
    else:
        results = []
        for fold in range(cfg.n_folds):
            results.append(train_one_fold(cfg, fold))

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        df = pd.DataFrame(results)
        print(df.mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
