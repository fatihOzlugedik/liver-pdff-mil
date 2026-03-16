"""
uncertainty.py — MC Dropout + Conformal Prediction + ECE
=========================================================
Usage (after training):
    python uncertainty.py \
        --exp_dir runs/mh_gated4_s384_f200 \
        --csv_path data/metadata.csv \
        --n_mc 50 \
        --coverage 0.90

Outputs (saved to exp_dir/uncertainty/):
    - mc_dropout_results.csv      : per-patient predictions + uncertainty
    - conformal_results.csv       : prediction intervals per patient
    - ece_reliability.png         : reliability diagram
    - uncertainty_summary.json    : ECE, coverage, mean interval width
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_absolute_error

# ── import from your existing src ──────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent / "src" / "src"))
from config import CFG, set_seed
from model import MILModel
from data import build_loaders


# ===========================================================================
# MC Dropout helpers
# ===========================================================================

def enable_dropout(model: nn.Module):
    """Set model to train mode only for Dropout layers (keeps BN in eval)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model: nn.Module,
                       loader: torch.utils.data.DataLoader,
                       device: torch.device,
                       n_mc: int = 50):
    """
    Run N stochastic forward passes per video.
    Returns:
        ids       : list[str]
        means     : np.ndarray (N_patients,)   — mean prediction
        stds      : np.ndarray (N_patients,)   — std (epistemic uncertainty)
        targets   : np.ndarray (N_patients,)
        all_preds : np.ndarray (N_patients, n_mc)
    """
    enable_dropout(model)

    ids, targets, all_preds = [], [], []

    with torch.no_grad():
        for bag, y, vid in loader:
            bag = bag.to(device)
            preds_i = []
            for _ in range(n_mc):
                y_hat = model(bag).item()
                preds_i.append(y_hat)
            ids.append(str(vid))
            targets.append(y.item())
            all_preds.append(preds_i)

    all_preds = np.array(all_preds)          # (N, n_mc)
    means     = all_preds.mean(axis=1)
    stds      = all_preds.std(axis=1)
    targets   = np.array(targets)

    return ids, means, stds, targets, all_preds


# ===========================================================================
# Conformal Prediction (split conformal, regression)
# ===========================================================================

def fit_conformal(cal_preds: np.ndarray,
                  cal_targets: np.ndarray,
                  coverage: float = 0.90):
    """
    Compute conformal threshold q from calibration set.
    Uses nonconformity score: |y_pred - y_true|
    Returns q (float) — add/subtract from test predictions to get intervals.
    """
    scores = np.abs(cal_preds - cal_targets)
    n = len(scores)
    # Finite-sample corrected quantile
    q_level = np.ceil((n + 1) * coverage) / n
    q_level = min(q_level, 1.0)
    q = np.quantile(scores, q_level)
    print(f"[Conformal] n_cal={n}, coverage={coverage}, q={q:.4f}")
    return float(q)


def apply_conformal(test_preds: np.ndarray, q: float):
    """Return (lower, upper) prediction intervals."""
    lower = test_preds - q
    upper = test_preds + q
    return lower, upper


def empirical_coverage(lower, upper, targets):
    """Fraction of targets inside [lower, upper]."""
    inside = ((targets >= lower) & (targets <= upper)).mean()
    return float(inside)


# ===========================================================================
# ECE + Reliability Diagram (binary: fatty/non-fatty)
# ===========================================================================

def compute_ece(probs: np.ndarray,
                labels: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Expected Calibration Error for binary classification.
    probs  : predicted probability of positive class (fatty)
    labels : true binary labels (0/1)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * np.abs(acc - conf)
    ece /= len(probs)
    return float(ece)


def plot_reliability_diagram(probs: np.ndarray,
                             labels: np.ndarray,
                             save_path: Path,
                             threshold: float = 5.0,
                             n_bins: int = 10):
    """
    Reliability diagram + confidence histogram.
    """
    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins)
    ece = compute_ece(probs, labels, n_bins)

    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    # ── reliability diagram ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax1.plot(mean_pred, fraction_pos, "o-", color="#2E5FA3", lw=2,
             markersize=7, label=f"Model (ECE={ece:.4f})")

    # shade gap between model and diagonal
    ax1.fill_between(mean_pred, mean_pred, fraction_pos,
                     alpha=0.15, color="#2E5FA3")

    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of positives (Fatty)", fontsize=12)
    ax1.set_title(f"Reliability Diagram — Binary Classification (threshold={threshold}%)",
                  fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xticklabels([])

    # ── confidence histogram ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(probs, bins=n_bins, range=(0, 1), color="#A8C8E0", edgecolor="white")
    ax2.set_xlabel("Mean predicted probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ECE] Reliability diagram saved → {save_path}")
    return ece


# ===========================================================================
# High-uncertainty case analysis
# ===========================================================================

def analyze_high_uncertainty(df: pd.DataFrame,
                              uncertainty_col: str = "mc_std",
                              percentile: float = 75.0,
                              save_path: Path = None):
    """
    Compare high-uncertainty vs low-uncertainty cases.
    Expects df to have: mc_std, target, prediction, and optionally BMI/PDFF columns.
    """
    threshold_val = np.percentile(df[uncertainty_col], percentile)
    df["high_uncertainty"] = df[uncertainty_col] >= threshold_val

    high = df[df["high_uncertainty"]]
    low  = df[~df["high_uncertainty"]]

    summary = {
        "uncertainty_threshold": float(threshold_val),
        "n_high": int(len(high)),
        "n_low":  int(len(low)),
        "mae_high": float(mean_absolute_error(high["target"], high["prediction"])),
        "mae_low":  float(mean_absolute_error(low["target"],  low["prediction"])),
        "mean_pdff_high": float(high["target"].mean()),
        "mean_pdff_low":  float(low["target"].mean()),
    }

    # If BMI column exists
    if "BMI" in df.columns:
        summary["mean_bmi_high"] = float(high["BMI"].mean())
        summary["mean_bmi_low"]  = float(low["BMI"].mean())

    print("\n[High-Uncertainty Analysis]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if save_path:
        # Plot: uncertainty vs MAE per patient
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(df[uncertainty_col], np.abs(df["target"] - df["prediction"]),
                        alpha=0.6, color="#2E5FA3", s=30)
        axes[0].axvline(threshold_val, color="red", linestyle="--", lw=1.5,
                        label=f"P{int(percentile)} threshold")
        axes[0].set_xlabel("MC Dropout Std (Uncertainty)", fontsize=12)
        axes[0].set_ylabel("Absolute Error (PDFF %)", fontsize=12)
        axes[0].set_title("Uncertainty vs Error", fontsize=13)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].boxplot([low[uncertainty_col].values, high[uncertainty_col].values],
                        labels=["Low uncertainty", "High uncertainty"],
                        patch_artist=True,
                        boxprops=dict(facecolor="#D5E8F0"))
        axes[1].set_ylabel("PDFF Target Value", fontsize=12)
        axes[1].set_title("PDFF Distribution by Uncertainty Group", fontsize=13)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Uncertainty] High-uncertainty plot saved → {save_path}")

    return summary


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir",   required=True,
                   help="Path to fold experiment dir (contains best.pt and fold subdirs)")
    p.add_argument("--csv_path",  required=True,
                   help="Metadata CSV (same as used in training)")
    p.add_argument("--n_mc",      type=int, default=50,
                   help="Number of MC Dropout forward passes")
    p.add_argument("--coverage",  type=float, default=0.90,
                   help="Conformal coverage level (e.g. 0.90)")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="PDFF threshold for binary classification (default 5.0)")
    p.add_argument("--n_folds",   type=int, default=5)
    p.add_argument("--aggregator", default="mh_gated4")
    p.add_argument("--img_size",  type=int, default=384)
    p.add_argument("--n_frames",  type=int, default=200)
    return p.parse_args()


def run_fold(fold_idx: int,
             exp_dir: Path,
             args,
             all_val_preds: list,
             all_val_targets: list,
             all_test_results: list):
    """Process one fold: MC Dropout on val (calibration) + test."""

    fold_dir = exp_dir / f"fold{fold_idx:02d}"
    ckpt     = fold_dir / "best.pt"
    if not ckpt.exists():
        print(f"[WARN] No checkpoint found at {ckpt}, skipping fold {fold_idx}")
        return

    # ── build config & loaders ─────────────────────────────────────────────
    cfg = CFG(
        aggregator = args.aggregator,
        img_size   = args.img_size,
        n_frames   = args.n_frames,
        n_folds    = args.n_folds,
        exp_dir    = fold_dir,
    )
    cfg.csv_path = args.csv_path
    set_seed(cfg.seed)

    _, val_dl, te_dl, _, val_df, tst_df = build_loaders(cfg, fold_idx)

    # ── load model ─────────────────────────────────────────────────────────
    model = MILModel(cfg).to(cfg.device)
    model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
    print(f"\n[Fold {fold_idx}] Loaded checkpoint: {ckpt}")

    # ── MC Dropout — validation (calibration) ──────────────────────────────
    print(f"[Fold {fold_idx}] MC Dropout on validation set (n_mc={args.n_mc})...")
    val_ids, val_means, val_stds, val_targets, _ = mc_dropout_predict(
        model, val_dl, cfg.device, n_mc=args.n_mc
    )
    all_val_preds.extend(val_means.tolist())
    all_val_targets.extend(val_targets.tolist())

    # ── Conformal threshold from validation ────────────────────────────────
    q = fit_conformal(val_means, val_targets, coverage=args.coverage)

    # ── MC Dropout — test ──────────────────────────────────────────────────
    print(f"[Fold {fold_idx}] MC Dropout on test set (n_mc={args.n_mc})...")
    te_ids, te_means, te_stds, te_targets, _ = mc_dropout_predict(
        model, te_dl, cfg.device, n_mc=args.n_mc
    )

    # ── Conformal intervals on test ────────────────────────────────────────
    lower, upper = apply_conformal(te_means, q)
    coverage_achieved = empirical_coverage(lower, upper, te_targets)
    print(f"[Fold {fold_idx}] Empirical coverage on test: {coverage_achieved:.3f} "
          f"(target: {args.coverage})")

    # ── Save fold test results ─────────────────────────────────────────────
    out_dir = fold_dir / "uncertainty"
    out_dir.mkdir(exist_ok=True)

    # Probability of fatty for ECE (sigmoid of regression output)
    te_probs  = torch.sigmoid(torch.tensor(te_means)).numpy()
    te_binary = (te_targets > args.threshold).astype(int)

    test_df_out = pd.DataFrame({
        "video_id":       te_ids,
        "target":         te_targets,
        "mc_mean":        te_means,
        "mc_std":         te_stds,
        "conformal_lower": lower,
        "conformal_upper": upper,
        "interval_width": upper - lower,
        "prob_fatty":     te_probs,
        "binary_true":    te_binary,
        "conformal_q":    q,
        "fold":           fold_idx,
    })
    test_df_out.to_csv(out_dir / "mc_conformal_test.csv", index=False)

    all_test_results.append(test_df_out)

    del model
    torch.cuda.empty_cache()


def main():
    args    = parse_args()
    exp_dir = Path(args.exp_dir)
    out_dir = exp_dir / "uncertainty"
    out_dir.mkdir(exist_ok=True)

    all_val_preds   = []
    all_val_targets = []
    all_test_results = []

    # ── Run per fold ────────────────────────────────────────────────────────
    for fold_idx in range(args.n_folds):
        run_fold(fold_idx, exp_dir, args,
                 all_val_preds, all_val_targets, all_test_results)

    if not all_test_results:
        print("[ERROR] No fold results collected. Check exp_dir and checkpoints.")
        return

    # ── Aggregate across folds ──────────────────────────────────────────────
    combined = pd.concat(all_test_results, ignore_index=True)
    combined.to_csv(out_dir / "mc_conformal_all_folds.csv", index=False)
    print(f"\n[Combined] {len(combined)} test patients across {args.n_folds} folds")

    # ── ECE + Reliability Diagram ───────────────────────────────────────────
    probs  = combined["prob_fatty"].values
    labels = combined["binary_true"].values
    ece    = plot_reliability_diagram(
        probs, labels,
        save_path = out_dir / "reliability_diagram.png",
        threshold = args.threshold,
    )

    # ── High-uncertainty analysis ───────────────────────────────────────────
    uncertainty_summary = analyze_high_uncertainty(
        combined,
        uncertainty_col = "mc_std",
        percentile      = 75.0,
        save_path       = out_dir / "uncertainty_analysis.png",
    )

    # ── Overall conformal coverage ──────────────────────────────────────────
    achieved_cov = empirical_coverage(
        combined["conformal_lower"].values,
        combined["conformal_upper"].values,
        combined["target"].values,
    )
    mean_width = float(combined["interval_width"].mean())

    # ── Global conformal fit (all val preds pooled) ─────────────────────────
    val_preds_arr   = np.array(all_val_preds)
    val_targets_arr = np.array(all_val_targets)
    global_q = fit_conformal(val_preds_arr, val_targets_arr, coverage=args.coverage)

    # ── Summary JSON ────────────────────────────────────────────────────────
    summary = {
        "n_folds":              args.n_folds,
        "n_mc_passes":          args.n_mc,
        "coverage_target":      args.coverage,
        "coverage_achieved":    achieved_cov,
        "mean_interval_width":  mean_width,
        "global_conformal_q":   global_q,
        "ECE":                  ece,
        "mae_overall":          float(mean_absolute_error(
                                    combined["target"], combined["mc_mean"])),
        "mean_mc_std":          float(combined["mc_std"].mean()),
        "high_uncertainty_analysis": uncertainty_summary,
    }
    with open(out_dir / "uncertainty_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("UNCERTAINTY SUMMARY")
    print("="*60)
    print(f"  ECE:                  {ece:.4f}")
    print(f"  Coverage (target):    {args.coverage:.2f}")
    print(f"  Coverage (achieved):  {achieved_cov:.4f}")
    print(f"  Mean interval width:  {mean_width:.4f} PDFF%")
    print(f"  Global conformal q:   {global_q:.4f}")
    print(f"  Mean MC std:          {combined['mc_std'].mean():.4f}")
    print(f"  Results saved to:     {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()



'''
python uncertainty.py \
    --exp_dir runs/mh_gated4_s384_f200 \
    --csv_path data/metadata.csv \
    --n_mc 50 \
    --coverage 0.90 \
    --aggregator mh_gated4 \
    --img_size 384 \
    --n_frames 200
'''