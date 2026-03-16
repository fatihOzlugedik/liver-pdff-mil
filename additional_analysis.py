#!/usr/bin/env python3
"""
additional_analysis.py — Clinical-grade extension analysis for the WACV paper.

Reads existing prediction and uncertainty CSVs from a fold directory and
produces publication-ready figures and a comprehensive metrics report.

Analyses implemented
────────────────────
1.  Regression: MAE, RMSE, Pearson r, Bias, limits-of-agreement
2.  Bland-Altman plot (primary clinical agreement figure)
3.  Binary classification (≥5%): AUROC + ROC curve, sensitivity, specificity,
    PPV, NPV, F1, balanced accuracy
4.  4-class classification (PDFF thresholds: 6.4 / 16.3 / 20.7%)
5.  Stratified MAE + confusion by PDFF severity grade
6.  High-fat bias analysis with regression trend
7.  Uncertainty integration: TTA σ as clinical triage flag
8.  Conformal prediction coverage + interval calibration
9.  Training convergence curve
10. Combined clinical summary figure (4-panel, paper-ready)

Usage
─────
  python additional_analysis.py --fold_dir runs_without_test/mh_gated4_s384_f175/fold00
  python additional_analysis.py --fold_dir /absolute/path/to/fold00
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    auc, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─── Clinical thresholds ──────────────────────────────────────────────────────
BINARY_THR  = 5.0                          # FDA/AASLD steatosis cutoff
GRADE_THRS  = [6.4, 16.3, 20.7]           # normal / mild / moderate / severe
GRADE_NAMES = ["Normal\n(<6.4%)", "Mild\n(6.4–16.3%)",
               "Moderate\n(16.3–20.7%)", "Severe\n(>20.7%)"]
GRADE_COLORS = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(fold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load predictions and (optionally) uncertainty metrics.

    val_predicted_true_values.csv  columns: video, stage, target, prediction
    uncertainty/uncertainty_metrics.csv  columns: video, gt_pdff, pred_pdff,
        mae, mcsd_std, tta_std, cp_lower, cp_upper, cp_covered, …
    """
    pred_csv = fold_dir / "val_predicted_true_values.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_csv}")

    preds = pd.read_csv(pred_csv)
    # normalise column names
    preds = preds.rename(columns={"target": "gt_pdff", "prediction": "pred_pdff"})
    preds["mae"] = (preds["pred_pdff"] - preds["gt_pdff"]).abs()

    unc_csv = fold_dir / "uncertainty" / "uncertainty_metrics.csv"
    unc = pd.read_csv(unc_csv) if unc_csv.exists() else None

    if unc is not None:
        # merge on video id (left join — keep all preds)
        preds = preds.merge(
            unc[["video", "mcsd_std", "tta_std", "cp_lower", "cp_upper", "cp_covered"]],
            on="video", how="left",
        )

    return preds, unc


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Grade assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_grade(pdff: pd.Series) -> pd.Series:
    """0=Normal, 1=Mild, 2=Moderate, 3=Severe."""
    g = pd.cut(pdff, bins=[-np.inf] + GRADE_THRS + [np.inf],
               labels=[0, 1, 2, 3]).astype(int)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 1 & 2. Regression + Bland-Altman
# ─────────────────────────────────────────────────────────────────────────────

def regression_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    errors = pred - gt
    r, p_r  = stats.pearsonr(gt, pred)
    slope, intercept, *_ = stats.linregress(gt, pred)
    return {
        "n":          len(gt),
        "mae":        np.abs(errors).mean(),
        "mae_std":    np.abs(errors).std(),
        "rmse":       np.sqrt((errors**2).mean()),
        "bias":       errors.mean(),
        "bias_std":   errors.std(),
        "loa_upper":  errors.mean() + 1.96 * errors.std(),
        "loa_lower":  errors.mean() - 1.96 * errors.std(),
        "r":          r,
        "p_r":        p_r,
        "slope":      slope,
        "intercept":  intercept,
        "within2":    (np.abs(errors) < 2).mean(),
        "within3":    (np.abs(errors) < 3).mean(),
        "within5":    (np.abs(errors) < 5).mean(),
    }


def plot_bland_altman(gt: np.ndarray, pred: np.ndarray,
                      m: dict, out_dir: Path) -> None:
    means  = (gt + pred) / 2
    diff   = pred - gt

    fig, ax = plt.subplots(figsize=(8, 6))

    # scatter coloured by severity grade
    grades = assign_grade(pd.Series(gt))
    for g, name, color in zip([0,1,2,3], GRADE_NAMES, GRADE_COLORS):
        mask = grades == g
        ax.scatter(means[mask], diff[mask], color=color, s=55,
                   edgecolors="white", linewidth=0.6, zorder=3,
                   label=name.replace("\n", " "))

    # Bias and LoA lines
    ax.axhline(m["bias"],       color="#1565C0", linewidth=1.8,
               linestyle="-",  label=f"Bias = {m['bias']:+.2f}%")
    ax.axhline(m["loa_upper"],  color="#1565C0", linewidth=1.2,
               linestyle="--", label=f"+1.96 SD = {m['loa_upper']:+.2f}%")
    ax.axhline(m["loa_lower"],  color="#1565C0", linewidth=1.2,
               linestyle="--", label=f"−1.96 SD = {m['loa_lower']:+.2f}%")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    # shaded LoA band
    ax.fill_between(ax.get_xlim() if ax.get_xlim() != (0, 1) else [0, 30],
                    m["loa_lower"], m["loa_upper"],
                    alpha=0.07, color="#1565C0")

    ax.set_xlabel("Mean of MRI-PDFF and Predicted PDFF (%)")
    ax.set_ylabel("Predicted − MRI-PDFF (%)")
    ax.set_title(
        f"Bland-Altman Plot  (n={m['n']})\n"
        f"Bias = {m['bias']:+.2f}%  |  95% LoA = [{m['loa_lower']:+.2f}%, "
        f"{m['loa_upper']:+.2f}%]",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    savefig(fig, out_dir, "fig01_bland_altman")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Binary classification — ROC + clinical metrics
# ─────────────────────────────────────────────────────────────────────────────

def binary_classification_metrics(gt: np.ndarray, pred: np.ndarray,
                                   thr: float = BINARY_THR) -> dict:
    gt_bin   = (gt   >= thr).astype(int)
    pred_bin = (pred >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(gt_bin, pred_bin).ravel()
    auroc = roc_auc_score(gt_bin, pred)
    fpr, tpr, _ = roc_curve(gt_bin, pred)
    return {
        "auroc":    auroc,
        "fpr":      fpr,
        "tpr":      tpr,
        "f1":       f1_score(gt_bin, pred_bin),
        "bal_acc":  balanced_accuracy_score(gt_bin, pred_bin),
        "sens":     tp / (tp + fn),
        "spec":     tn / (tn + fp),
        "ppv":      tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv":      tn / (tn + fn) if (tn + fn) > 0 else 0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_pos":    int(gt_bin.sum()),
        "n_neg":    int((1 - gt_bin).sum()),
        "conf_mat": confusion_matrix(gt_bin, pred_bin),
    }


def plot_roc(bm: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) ROC curve
    ax = axes[0]
    ax.plot(bm["fpr"], bm["tpr"], color="#1565C0", linewidth=2.5,
            label=f"AUROC = {bm['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.fill_between(bm["fpr"], bm["tpr"], alpha=0.08, color="#1565C0")
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(f"ROC Curve — Binary Steatosis (PDFF ≥ {BINARY_THR}%)",
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # (b) Clinical metrics bar
    ax = axes[1]
    metrics = {
        "Sensitivity": bm["sens"],
        "Specificity": bm["spec"],
        "PPV":         bm["ppv"],
        "NPV":         bm["npv"],
        "F1":          bm["f1"],
        "Bal. Acc.":   bm["bal_acc"],
        "AUROC":       bm["auroc"],
    }
    colors = ["#1565C0" if v >= 0.80 else "#FFA726" if v >= 0.70
              else "#EF5350" for v in metrics.values()]
    bars = ax.barh(list(metrics.keys()), list(metrics.values()),
                   color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, metrics.values()):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.axvline(0.80, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.6, label="0.80 target")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score")
    ax.set_title(
        f"Clinical Metrics  (n={bm['n_pos']+bm['n_neg']}: "
        f"{bm['n_pos']} steatotic / {bm['n_neg']} normal)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    savefig(fig, out_dir, "fig02_roc_clinical_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. 4-class classification + stratified MAE
# ─────────────────────────────────────────────────────────────────────────────

def plot_4class_analysis(gt: np.ndarray, pred: np.ndarray,
                         out_dir: Path) -> None:
    gt_g   = assign_grade(pd.Series(gt)).values
    pred_g = assign_grade(pd.Series(pred)).values

    cm = confusion_matrix(gt_g, pred_g, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    f1_w = f1_score(gt_g, pred_g, average="weighted", zero_division=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Confusion matrix
    ax = axes[0]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    short = ["Normal", "Mild", "Moderate", "Severe"]
    ax.set_xticks(range(4)); ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted Grade")
    ax.set_ylabel("True Grade (MRI-PDFF)")
    ax.set_title(f"4-Class Confusion Matrix\n(row-normalised  |  wF1={f1_w:.2f})",
                 fontweight="bold")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i,j] > 0.55 else "black")

    # (b) Stratified MAE
    ax = axes[1]
    errors_by_grade = []
    ns = []
    for g in range(4):
        mask = gt_g == g
        ns.append(mask.sum())
        errors_by_grade.append(np.abs(pred[mask] - gt[mask]) if mask.sum() > 0
                                else np.array([0.0]))
    means = [e.mean() for e in errors_by_grade]
    stds  = [e.std()  for e in errors_by_grade]

    bars = ax.bar(range(4), means, yerr=stds, color=GRADE_COLORS,
                  edgecolor="white", capsize=6, width=0.55, alpha=0.85)
    for i, (bar, n, m) in enumerate(zip(bars, ns, means)):
        ax.text(bar.get_x() + bar.get_width()/2, m + stds[i] + 0.1,
                f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels([n.replace("\n", "\n") for n in GRADE_NAMES],
                       fontsize=9)
    ax.set_ylabel("MAE (PDFF %)")
    ax.set_title("Stratified MAE by PDFF Severity Grade",
                 fontweight="bold")

    # (c) Scatter by grade
    ax = axes[2]
    for g in range(4):
        mask = gt_g == g
        ax.scatter(gt[mask], pred[mask], color=GRADE_COLORS[g],
                   s=55, edgecolors="white", linewidth=0.6, zorder=3,
                   label=f"{short[g]} (n={mask.sum()})")
    lim = max(gt.max(), pred.max()) + 3
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("MRI-PDFF (%)")
    ax.set_ylabel("Predicted PDFF (%)")
    ax.set_title("Prediction vs MRI-PDFF\n(by PDFF grade)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-1, lim); ax.set_ylim(-1, lim)
    ax.set_aspect("equal")

    plt.tight_layout()
    savefig(fig, out_dir, "fig03_4class_stratified")


# ─────────────────────────────────────────────────────────────────────────────
# 6. High-fat bias analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_bias_analysis(gt: np.ndarray, pred: np.ndarray, out_dir: Path) -> None:
    errors = pred - gt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Error vs GT with LOESS-style rolling trend
    ax = axes[0]
    sort_idx = np.argsort(gt)
    gt_s, err_s = gt[sort_idx], errors[sort_idx]

    gt_g = assign_grade(pd.Series(gt)).values
    for g in range(4):
        mask = gt_g == g
        ax.scatter(gt[mask], errors[mask], color=GRADE_COLORS[g],
                   s=55, edgecolors="white", linewidth=0.6, zorder=3,
                   label=GRADE_NAMES[g].replace("\n", " "))

    # linear trend line
    slope, intercept, r, p, _ = stats.linregress(gt, errors)
    x_line = np.linspace(gt.min(), gt.max(), 100)
    ax.plot(x_line, slope * x_line + intercept,
            color="#C62828", linewidth=2, linestyle="--",
            label=f"Trend (r={r:.2f}, p={p:.3f})")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("MRI-PDFF (%)")
    ax.set_ylabel("Prediction Error (Pred − GT, PDFF %)")
    ax.set_title("Systematic Bias vs PDFF Level\n"
                 "(negative = underestimation)", fontweight="bold")
    ax.legend(fontsize=8)

    # Annotate mean bias per group
    for g, color in zip(range(4), GRADE_COLORS):
        mask = gt_g == g
        if mask.sum() > 0:
            mean_bias = errors[mask].mean()
            mean_gt   = gt[mask].mean()
            ax.annotate(f"{mean_bias:+.1f}%",
                        xy=(mean_gt, mean_bias),
                        fontsize=9, color=color, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")

    # (b) Box plot of errors by grade
    ax = axes[1]
    data_by_grade = [errors[gt_g == g] for g in range(4)]
    bp = ax.boxplot(data_by_grade, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], GRADE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticklabels([n.replace("\n", "\n") for n in GRADE_NAMES], fontsize=9)
    ax.set_ylabel("Prediction Error (Pred − GT, PDFF %)")
    ax.set_title("Error Distribution by PDFF Grade\n"
                 "(whiskers = 1.5×IQR)", fontweight="bold")

    # Annotate n and mean
    for i, (g, d) in enumerate(zip(range(4), data_by_grade), 1):
        ax.text(i, ax.get_ylim()[1] * 0.92,
                f"n={len(d)}\nμ={d.mean():+.1f}",
                ha="center", fontsize=8)

    plt.tight_layout()
    savefig(fig, out_dir, "fig04_bias_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Uncertainty as clinical triage flag
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_triage(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Triage analysis pivoted to binary classification errors (0/1).

    Key clinical question: at what TTA σ threshold can we auto-report
    predictions with ZERO false negatives (no missed steatosis patients)?

    Critical finding: FN cases have LOW TTA σ (model is confidently wrong).
    FP cases have HIGH TTA σ (model is appropriately uncertain on over-calls).
    This asymmetry means TTA cannot guarantee zero FN via high-σ flagging.
    Instead, a LOW-σ threshold is needed: flag cases the model is too
    confident about as a complementary safety net.
    """
    if "tta_std" not in df.columns:
        print("  [SKIP] Uncertainty columns not found — skipping triage plot.")
        return

    TRIAGE_THR = 1.0   # standard high-uncertainty flag

    gt   = df["gt_pdff"].values
    pred = df["pred_pdff"].values
    mae  = df["mae"].values
    tta  = df["tta_std"].values

    gt_bin   = (gt   >= BINARY_THR).astype(int)
    pred_bin = (pred >= BINARY_THR).astype(int)
    is_fn    = (gt_bin == 1) & (pred_bin == 0)   # missed steatosis
    is_fp    = (gt_bin == 0) & (pred_bin == 1)   # over-called
    is_error = is_fn | is_fp

    n_fn  = int(is_fn.sum())
    n_fp  = int(is_fp.sum())
    n_err = int(is_error.sum())

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    np.random.seed(42)

    # ── (a) TTA σ per outcome — key finding: FN have LOW σ ──────────────────
    ax = axes[0]
    groups = {
        "True Negative\n(correct normal)":    (~is_error) & (gt_bin == 0),
        "True Positive\n(correct steatosis)":  (~is_error) & (gt_bin == 1),
        "False Positive\n(over-called)":        is_fp,
        "False Negative\n(missed patient)":     is_fn,
    }
    colors_grp = ["#4CAF50", "#1565C0", "#FFA726", "#F44336"]
    positions, ticks = [], []
    for i, (label, mask) in enumerate(groups.items()):
        vals = tta[mask]
        if len(vals) == 0:
            continue
        bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker="o", markersize=5,
                                        markerfacecolor=colors_grp[i],
                                        markeredgecolor="white"))
        bp["boxes"][0].set_facecolor(colors_grp[i])
        bp["boxes"][0].set_alpha(0.75)
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, color=colors_grp[i], s=35,
                   alpha=0.7, zorder=4, edgecolors="white", linewidth=0.4)
        positions.append(i)
        ticks.append(f"{label}\n(n={mask.sum()})")

    ax.axhline(TRIAGE_THR, color="#C62828", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"High-σ flag (σ ≥ {TRIAGE_THR:.1f})")
    ax.set_xticks(positions)
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylabel("TTA Uncertainty (σ, PDFF %)")
    ax.set_title("TTA σ by Binary Classification Outcome\n"
                 "⚠ Missed patients (FN) have LOW uncertainty",
                 fontweight="bold")
    ax.legend(fontsize=8)

    # Annotate FN max σ — the critical boundary
    fn_max_sigma = tta[is_fn].max() if n_fn > 0 else 0
    ax.axhline(fn_max_sigma, color="#F44336", linestyle=":", linewidth=1.2,
               alpha=0.7)
    ax.text(len(positions) - 0.5, fn_max_sigma + 0.04,
            f"FN max σ = {fn_max_sigma:.2f}", fontsize=8,
            color="#C62828", ha="right")

    # ── (b) Operating curve: threshold sweep — FN-recall vs burden ───────────
    # Sweep in both directions:
    #   HIGH threshold (σ ≥ thr): standard "flag uncertain" — misses FN
    #   LOW threshold  (σ ≤ thr): flag confident predictions as safety net
    ax = axes[1]
    thresholds = np.linspace(tta.min(), tta.max(), 300)

    # High-σ flagging: flag if σ ≥ thr
    ff_high, fn_high, err_high = [], [], []
    # Low-σ flagging: flag if σ ≤ thr (safety net for confident-wrong cases)
    ff_low, fn_low = [], []

    for thr in thresholds:
        # High-σ: uncertain cases reviewed
        flagged_h = tta >= thr
        ff_high.append(flagged_h.mean())
        fn_high.append((flagged_h & is_fn).sum() / max(n_fn, 1))
        err_high.append((flagged_h & is_error).sum() / max(n_err, 1))

        # Low-σ: overly-confident cases reviewed
        flagged_l = tta <= thr
        ff_low.append(flagged_l.mean())
        fn_low.append((flagged_l & is_fn).sum() / max(n_fn, 1))

    ff_high = np.array(ff_high); fn_high = np.array(fn_high)
    err_high = np.array(err_high)
    ff_low  = np.array(ff_low);  fn_low  = np.array(fn_low)

    ax.plot(ff_high, fn_high, color="#F44336", linewidth=2,
            label="FN recall — flag high σ (standard)")
    ax.plot(ff_high, err_high, color="#1565C0", linewidth=1.5, linestyle="--",
            label="All errors — flag high σ")
    ax.plot(ff_low,  fn_low,  color="#E65100", linewidth=2, linestyle="-.",
            label="FN recall — flag LOW σ (safety net)")

    # Mark operating points
    # 1. Standard σ ≥ 1.0
    ff1  = (tta >= TRIAGE_THR).mean()
    fn1  = ((tta >= TRIAGE_THR) & is_fn).sum() / max(n_fn, 1)
    ax.scatter([ff1], [fn1], color="#F44336", s=130, zorder=5,
               edgecolors="black", linewidth=0.8,
               label=f"σ ≥ {TRIAGE_THR:.1f}: flag {ff1:.0%}, FN caught={fn1:.0%}")

    # 2. Zero-FN via low-σ: flag everything up to FN max σ
    zero_fn_low_thr = fn_max_sigma
    ff_zfn = (tta <= zero_fn_low_thr).mean()
    ax.scatter([ff_zfn], [1.0], color="#E65100", s=130, zorder=5,
               edgecolors="black", linewidth=0.8,
               label=f"σ ≤ {zero_fn_low_thr:.2f}: flag {ff_zfn:.0%}, FN caught=100%")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="100% FN recall (zero missed)")
    ax.set_xlabel("Fraction of Cases Flagged for Review")
    ax.set_ylabel("Fraction of Missed Patients Caught (FN Recall)")
    ax.set_title("Triage Operating Curve\n(goal: 100% FN recall with min burden)",
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.15)

    # ── (c) Stacked bar: patient composition in each pool ────────────────────
    # Show three strategies side by side
    ax = axes[2]
    strategies = {
        f"No triage\n(review all)\nn=50":
            np.ones(len(gt), dtype=bool),
        f"Flag high σ\n(σ ≥ {TRIAGE_THR:.1f})\nn={(tta>=TRIAGE_THR).sum()} reviewed":
            tta >= TRIAGE_THR,
        f"Flag low σ\n(σ ≤ {zero_fn_low_thr:.2f})\nn={(tta<=zero_fn_low_thr).sum()} reviewed":
            tta <= zero_fn_low_thr,
    }

    grp_masks  = [
        (~is_error) & (gt_bin == 0),   # TN
        (~is_error) & (gt_bin == 1),   # TP
        is_fp,                          # FP
        is_fn,                          # FN
    ]
    grp_labels = ["True Negative", "True Positive",
                  "False Positive", "False Negative\n(missed)"]
    grp_colors = ["#4CAF50", "#1565C0", "#FFA726", "#F44336"]

    x_pos = np.arange(len(strategies))
    bottoms = np.zeros(len(strategies))

    for gmask, glabel, gcolor in zip(grp_masks, grp_labels, grp_colors):
        vals = np.array([gmask[flag].sum() for flag in strategies.values()])
        bars = ax.bar(x_pos, vals, bottom=bottoms, color=gcolor, alpha=0.85,
                      edgecolor="white", linewidth=0.8, label=glabel)
        for xi, (bar, v) in enumerate(zip(bars, vals)):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottoms[xi] + v / 2, str(int(v)),
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white")
        bottoms += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(strategies.keys()), fontsize=8)
    ax.set_ylabel("Number of Patients in Flagged Pool")
    ax.set_title("Composition of Flagged Pool\nby Triage Strategy",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # Annotate FN count in each flagged pool
    for xi, flag in enumerate(strategies.values()):
        fn_in_pool = is_fn[flag].sum()
        safety = "✓ zero FN" if fn_in_pool == 0 else f"⚠ {fn_in_pool} FN"
        color  = "#2E7D32" if fn_in_pool == 0 else "#C62828"
        ax.text(xi, bottoms[xi] + 0.4, safety,
                ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    savefig(fig, out_dir, "fig05_uncertainty_triage")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Conformal prediction coverage
# ─────────────────────────────────────────────────────────────────────────────

def plot_conformal_summary(df: pd.DataFrame, out_dir: Path) -> None:
    if "cp_lower" not in df.columns:
        print("  [SKIP] CP columns not found — skipping conformal plot.")
        return

    gt        = df["gt_pdff"].values
    pred      = df["pred_pdff"].values
    cp_lower  = df["cp_lower"].values
    cp_upper  = df["cp_upper"].values
    covered   = df["cp_covered"].values.astype(bool)
    half_width = (cp_upper - cp_lower).mean() / 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Pred ± interval vs GT, sorted by GT
    ax = axes[0]
    sort_idx = np.argsort(gt)
    xs       = np.arange(len(gt))

    ax.scatter(xs, gt[sort_idx], color="black", s=40, zorder=5,
               label="MRI-PDFF (GT)", marker="D")
    ax.errorbar(xs, pred[sort_idx],
                yerr=[pred[sort_idx] - cp_lower[sort_idx],
                      cp_upper[sort_idx] - pred[sort_idx]],
                fmt="o", color="#1565C0", ecolor="#1565C0",
                elinewidth=0.8, capsize=2, markersize=4, alpha=0.7,
                label=f"Pred ± CP interval")

    # Highlight missed cases
    miss_idx = np.where(~covered[sort_idx])[0]
    ax.scatter(miss_idx, gt[sort_idx][miss_idx], color="#F44336",
               s=100, zorder=6, marker="X", label="Not covered")

    ax.set_xlabel("Patient (sorted by GT PDFF)")
    ax.set_ylabel("PDFF (%)")
    ax.set_title(
        f"Conformal Prediction Intervals\n"
        f"Half-width = ±{half_width:.1f}%  |  "
        f"Coverage = {covered.mean():.1%}  (n={len(gt)})",
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    # (b) Coverage by severity grade
    ax = axes[1]
    gt_g = assign_grade(pd.Series(gt)).values
    cov_by_grade = []
    ns = []
    for g in range(4):
        mask = gt_g == g
        ns.append(mask.sum())
        cov_by_grade.append(covered[mask].mean() if mask.sum() > 0 else 0.0)

    bars = ax.bar(range(4), cov_by_grade, color=GRADE_COLORS,
                  edgecolor="white", width=0.55, alpha=0.85)
    ax.axhline(0.90, color="#1565C0", linestyle="--", linewidth=1.5,
               label="90% nominal")
    for bar, n, cov in zip(bars, ns, cov_by_grade):
        ax.text(bar.get_x() + bar.get_width()/2,
                min(cov + 0.02, 1.0),
                f"{cov:.0%}\n(n={n})", ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels([n.replace("\n", "\n") for n in GRADE_NAMES], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("CP Coverage by PDFF Grade\n(90% nominal target)",
                 fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    savefig(fig, out_dir, "fig06_conformal_coverage")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Training convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(fold_dir: Path, out_dir: Path) -> None:
    log_csv = fold_dir / "train_log.csv"
    if not log_csv.exists():
        print("  [SKIP] train_log.csv not found.")
        return

    log = pd.read_csv(log_csv)
    best_epoch = log["val_mae"].idxmin()
    best_val   = log["val_mae"].min()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(log["epoch"], log["train_loss"], color="#1565C0",
            linewidth=1.5, label="Train L1 Loss")
    ax.plot(log["epoch"], log["val_mae"],   color="#F44336",
            linewidth=1.5, label="Val MAE")
    ax.axvline(log.loc[best_epoch, "epoch"], color="gray",
               linestyle="--", linewidth=1, alpha=0.7)
    ax.scatter([log.loc[best_epoch, "epoch"]], [best_val],
               color="#F44336", s=80, zorder=5,
               label=f"Best val MAE = {best_val:.3f}% (ep {log.loc[best_epoch,'epoch']:.0f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MAE (PDFF %)")
    ax.set_title("Training Convergence", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(log["epoch"], log["lr"], color="#2E7D32", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine LR Schedule", fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(log["epoch"], log["gpu_mb"] / 1024, color="#B0BEC5",
             linewidth=1, alpha=0.6)
    ax2.set_ylabel("GPU Memory (GB)", color="#B0BEC5")
    ax2.tick_params(axis="y", labelcolor="#B0BEC5")

    plt.tight_layout()
    savefig(fig, out_dir, "fig07_training_curve")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Combined 4-panel clinical summary (paper figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_clinical_summary(gt: np.ndarray, pred: np.ndarray,
                          bm: dict, m: dict,
                          df: pd.DataFrame, out_dir: Path) -> None:
    """
    Single publication-ready 4-panel figure:
      A  Scatter pred vs GT with identity line (by grade)
      B  Bland-Altman
      C  ROC curve
      D  Stratified MAE
    """
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    gt_g = assign_grade(pd.Series(gt)).values
    short = ["Normal", "Mild", "Moderate", "Severe"]

    # ── A: Scatter ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for g in range(4):
        mask = gt_g == g
        ax.scatter(gt[mask], pred[mask], color=GRADE_COLORS[g],
                   s=55, edgecolors="white", linewidth=0.6, zorder=3,
                   label=f"{short[g]} (n={mask.sum()})")
    lim = max(gt.max(), pred.max()) + 3
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.9, alpha=0.5)
    # regression line
    slope, intercept, r, p_r, _ = stats.linregress(gt, pred)
    x_l = np.linspace(0, lim, 100)
    ax.plot(x_l, slope * x_l + intercept, color="#1565C0",
            linewidth=1.5, linestyle="-", alpha=0.6)
    ax.set_xlim(-1, lim); ax.set_ylim(-1, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("MRI-PDFF (%)")
    ax.set_ylabel("Predicted PDFF (%)")
    ax.set_title(f"(A) Predicted vs MRI-PDFF\nr = {r:.3f}  |  "
                 f"MAE = {m['mae']:.2f} ± {m['mae_std']:.2f}%",
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.text(0.97, 0.05, f"Within 3%: {m['within3']:.0%}",
            transform=ax.transAxes, ha="right", fontsize=9, color="#1565C0")

    # ── B: Bland-Altman ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    means = (gt + pred) / 2
    diff  = pred - gt
    for g in range(4):
        mask = gt_g == g
        ax.scatter(means[mask], diff[mask], color=GRADE_COLORS[g],
                   s=55, edgecolors="white", linewidth=0.6, zorder=3,
                   label=short[g])
    ax.axhline(m["bias"],       color="#1565C0", linewidth=1.8, linestyle="-",
               label=f"Bias {m['bias']:+.2f}%")
    ax.axhline(m["loa_upper"],  color="#1565C0", linewidth=1.2, linestyle="--",
               label=f"+1.96 SD {m['loa_upper']:+.2f}%")
    ax.axhline(m["loa_lower"],  color="#1565C0", linewidth=1.2, linestyle="--",
               label=f"−1.96 SD {m['loa_lower']:+.2f}%")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Mean of MRI-PDFF and Predicted PDFF (%)")
    ax.set_ylabel("Pred − GT (PDFF %)")
    ax.set_title(f"(B) Bland-Altman Agreement\nBias = {m['bias']:+.2f}%  |  "
                 f"95% LoA [{m['loa_lower']:+.2f}%, {m['loa_upper']:+.2f}%]",
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")

    # ── C: ROC ───────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(bm["fpr"], bm["tpr"], color="#1565C0", linewidth=2.5,
            label=f"AUC = {bm['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.fill_between(bm["fpr"], bm["tpr"], alpha=0.08, color="#1565C0")
    # Mark operating point (current threshold)
    # find point closest to top-left corner
    dist  = np.sqrt(bm["fpr"]**2 + (1 - bm["tpr"])**2)
    opt   = np.argmin(dist)
    ax.scatter([bm["fpr"][opt]], [bm["tpr"][opt]], s=80, color="#F44336",
               zorder=5, label=f"Op. point\nSens={bm['sens']:.2f} / Spec={bm['spec']:.2f}")
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"(C) ROC Curve — Steatosis (PDFF ≥ {BINARY_THR}%)\n"
                 f"AUROC = {bm['auroc']:.3f}  |  "
                 f"F1 = {bm['f1']:.3f}  |  Bal. Acc = {bm['bal_acc']:.3f}",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    # ── D: Stratified MAE ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    means_g = [np.abs(pred[gt_g==g] - gt[gt_g==g]).mean()
               if (gt_g==g).sum()>0 else 0 for g in range(4)]
    stds_g  = [np.abs(pred[gt_g==g] - gt[gt_g==g]).std()
               if (gt_g==g).sum()>1 else 0 for g in range(4)]
    ns_g    = [(gt_g==g).sum() for g in range(4)]

    bars = ax.bar(range(4), means_g, yerr=stds_g,
                  color=GRADE_COLORS, edgecolor="white",
                  capsize=7, width=0.55, alpha=0.85)
    for i, (bar, n, mean) in enumerate(zip(bars, ns_g, means_g)):
        ax.text(bar.get_x() + bar.get_width()/2,
                mean + stds_g[i] + 0.15,
                f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels([n.replace("\n", "\n") for n in GRADE_NAMES], fontsize=9)
    ax.set_ylabel("MAE (PDFF %)")
    ax.set_title("(D) MAE by PDFF Severity Grade\n(mean ± SD)",
                 fontweight="bold")
    ax.axhline(m["mae"], color="black", linestyle="--", linewidth=1,
               alpha=0.5, label=f"Overall MAE = {m['mae']:.2f}%")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Clinical Performance Summary — Swin-Tiny + Multi-Head Gated MIL\n"
        f"(n = {m['n']} validation videos, MRI-PDFF as reference standard)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    savefig(fig, out_dir, "fig00_clinical_summary_PAPER")


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def save_report(m: dict, bm: dict, df: pd.DataFrame,
                fold_dir: Path, out_dir: Path) -> None:
    gt   = df["gt_pdff"].values
    pred = df["pred_pdff"].values
    gt_g = assign_grade(pd.Series(gt)).values
    pred_g = assign_grade(pd.Series(pred)).values

    f1_4cls = f1_score(gt_g, pred_g, average="weighted", zero_division=0)
    ba_4cls = balanced_accuracy_score(gt_g, pred_g)

    tta_line = ""
    if "tta_std" in df.columns:
        r_tta, p_tta = stats.pearsonr(df["tta_std"].values, df["mae"].values)
        tta_line = (f"\n  TTA σ vs MAE        : r = {r_tta:.3f}  "
                    f"(p = {p_tta:.4f})")
        if "cp_covered" in df.columns:
            tta_line += (f"\n  CP empirical cov.   : "
                         f"{df['cp_covered'].mean():.1%}")

    lines = [
        "=" * 70,
        "ADDITIONAL CLINICAL ANALYSIS — FULL METRICS REPORT",
        "=" * 70,
        f"\nFold directory : {fold_dir}",
        f"N (val set)    : {m['n']}",
        "",
        "─── Regression ────────────────────────────────────────────────────",
        f"  MAE           : {m['mae']:.3f} ± {m['mae_std']:.3f} PDFF%",
        f"  RMSE          : {m['rmse']:.3f} PDFF%",
        f"  Pearson r     : {m['r']:.3f}  (p = {m['p_r']:.2e})",
        f"  Bias (Pred-GT): {m['bias']:+.3f} ± {m['bias_std']:.3f} PDFF%",
        f"  95% LoA       : [{m['loa_lower']:+.3f}%,  {m['loa_upper']:+.3f}%]",
        f"  Within ±2%    : {m['within2']:.1%}  ({int(m['within2']*m['n'])}/{m['n']})",
        f"  Within ±3%    : {m['within3']:.1%}  ({int(m['within3']*m['n'])}/{m['n']})",
        f"  Within ±5%    : {m['within5']:.1%}  ({int(m['within5']*m['n'])}/{m['n']})",
        "",
        f"─── Binary Classification (PDFF ≥ {BINARY_THR}%) ─────────────────────────",
        f"  AUROC         : {bm['auroc']:.3f}",
        f"  F1            : {bm['f1']:.3f}",
        f"  Balanced Acc. : {bm['bal_acc']:.3f}",
        f"  Sensitivity   : {bm['sens']:.3f}  ({bm['tp']}/{bm['tp']+bm['fn']} steatotic)",
        f"  Specificity   : {bm['spec']:.3f}  ({bm['tn']}/{bm['tn']+bm['fp']} normal)",
        f"  PPV           : {bm['ppv']:.3f}",
        f"  NPV           : {bm['npv']:.3f}",
        f"  Confusion     : TP={bm['tp']}  FP={bm['fp']}  FN={bm['fn']}  TN={bm['tn']}",
        "",
        "─── 4-Class Classification (6.4 / 16.3 / 20.7% thresholds) ────────",
        f"  Weighted F1   : {f1_4cls:.3f}",
        f"  Balanced Acc. : {ba_4cls:.3f}",
    ]

    for g in range(4):
        mask = gt_g == g
        if mask.sum() > 0:
            mae_g = np.abs(pred[mask] - gt[mask]).mean()
            bias_g = (pred[mask] - gt[mask]).mean()
            lines.append(f"  {['Normal','Mild','Moderate','Severe'][g]:<10}: "
                         f"n={mask.sum():2d}  MAE={mae_g:.2f}%  "
                         f"Bias={bias_g:+.2f}%")

    lines += [
        "",
        "─── Uncertainty (if available) ──────────────────────────────────",
        tta_line,
        "",
        "─── WACV Paper Reference (5-fold CV, test set) ──────────────────",
        "  MAE (reported)  : 4.10 [3.26–4.93]%",
        "  F1  (reported)  : 0.82",
        "  Bal. Acc.       : 0.81",
        "  Note: above metrics use train+val (no held-out test) → "
        "optimistic vs paper",
        "",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print(report)
    (out_dir / "clinical_metrics_report.txt").write_text(report)
    print(f"\n  Saved: clinical_metrics_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold_dir", required=True, type=Path,
                   help="Path to fold directory containing "
                        "val_predicted_true_values.csv")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    fold_dir = args.fold_dir.resolve()
    out_dir  = fold_dir / "additional_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFold dir : {fold_dir}")
    print(f"Output   : {out_dir}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    df, unc = load_data(fold_dir)
    gt   = df["gt_pdff"].values
    pred = df["pred_pdff"].values
    print(f"Loaded {len(df)} validation cases.\n")

    # ── Core metrics ──────────────────────────────────────────────────────────
    m  = regression_metrics(gt, pred)
    bm = binary_classification_metrics(gt, pred)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures …")

    print("\n[1/9] Bland-Altman")
    plot_bland_altman(gt, pred, m, out_dir)

    print("[2/9] ROC + clinical metrics")
    plot_roc(bm, out_dir)

    print("[3/9] 4-class + stratified MAE")
    plot_4class_analysis(gt, pred, out_dir)

    print("[4/9] Bias analysis")
    plot_bias_analysis(gt, pred, out_dir)

    print("[5/9] Uncertainty triage")
    plot_uncertainty_triage(df, out_dir)

    print("[6/9] Conformal coverage")
    plot_conformal_summary(df, out_dir)

    print("[7/9] Training curve")
    plot_training_curve(fold_dir, out_dir)

    print("[8/9] Clinical summary (paper figure)")
    plot_clinical_summary(gt, pred, bm, m, df, out_dir)

    print("[9/9] Text report")
    save_report(m, bm, df, fold_dir, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")
    print("\nFigure index:")
    print("  fig00_clinical_summary_PAPER.png  ← 4-panel paper figure")
    print("  fig01_bland_altman.png")
    print("  fig02_roc_clinical_metrics.png")
    print("  fig03_4class_stratified.png")
    print("  fig04_bias_analysis.png")
    print("  fig05_uncertainty_triage.png")
    print("  fig06_conformal_coverage.png")
    print("  fig07_training_curve.png")
    print("  clinical_metrics_report.txt")


if __name__ == "__main__":
    main()
