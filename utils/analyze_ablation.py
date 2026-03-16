"""
analyze_ablation.py
====================
Reads all ablation results under the runs folder and visualizes them.

Run folder name format: {aggregator}_s{img_size}_f{n_frames}/fold{XX}/
Per fold:
  - train_log.csv  → best val_mae
  - config.json    → img_size, n_frames, aggregator, window_size
Optional (if present):
  - predictions CSV (e.g., preds.csv / predictions.csv / *pred*.csv)
    with columns: patient_ID, target, prediction (+ optional set column)

Generated figures:
  1. Bar chart: aggregator × setup, mean Val MAE ± 95% CI
  2. Box plot: per aggregator, colored by setup
  3. Heatmap: aggregator × setup → mean Val MAE
  4. Learning curves: per aggregator (all folds, all setups)
  5. Scatter: final train loss vs best val MAE (overfitting)
  6. Setup effect (lines): per aggregator, mean Val MAE ± 95% CI across setups
  7. Confusion matrix grids (Binary thr=5.0): rows=aggregator, cols=fold (per setup)
  8. Confusion matrix grids (4-class): rows=aggregator, cols=fold (per setup)
  9. Bar chart: aggregator × setup, mean Binary F1 ± 95% CI (if preds CSV exists)
 10. Bar chart: aggregator × setup, mean 4-class weighted F1 ± 95% CI (if preds CSV exists)
 11. Setup effect (lines): mean Binary F1 ± 95% CI across setups (if available)
 12. Setup effect (lines): mean 4-class weighted F1 ± 95% CI across setups (if available)

Usage:
    python utils/analyze_ablation.py
    python utils/analyze_ablation.py --runs_dir /path/to/runs --out_dir ./figs
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats

from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RUNS_DIR = Path("/research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/runs_without_test")
OUT_DIR  = Path("/research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/runs_without_test/ablation_figures")

PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#607D8B", "#E91E63", "#3F51B5",
]

BINARY_THR = 5.0  # PDFF threshold for binary split: <=thr vs >thr

CLASS4_CUTS = (6.4, 16.3, 20.7)  # 4-class boundaries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ci95(values):
    """95% CI half-width via t-distribution."""
    n = len(values)
    if n < 2:
        return 0.0
    se = stats.sem(values)
    return float(se * stats.t.ppf(0.975, df=n - 1))


def parse_run_dir(run_dir: Path):
    """
    Parse run folder name to extract aggregator, img_size, n_frames.
    Expected format: {agg}_s{size}_f{frames}
    """
    name = run_dir.name  # e.g. mh_gated4_s224_f600
    m_size   = re.search(r"_s(\d+)", name)
    m_frames = re.search(r"_f(\d+)", name)
    img_size = int(m_size.group(1))   if m_size   else None
    n_frames = int(m_frames.group(1)) if m_frames else None

    # aggregator = everything before _s{size}
    agg_part = name[:m_size.start()] if m_size else name
    agg_part = agg_part.strip("_")

    return agg_part, img_size, n_frames


def find_preds_csv(fold_dir: Path, run_dir: Path) -> Path | None:
    """
    Try to locate a predictions CSV for a fold.
    Expected columns: patient_ID, target, prediction (+ optional set column).
    Searches fold_dir first, then run_dir.
    """
    # Common explicit names
    for name in ["preds.csv", "predictions.csv", "test_preds.csv", "test_predictions.csv"]:
        p = fold_dir / name
        if p.exists():
            return p

    # Heuristic search
    candidates = []
    for root in [fold_dir, run_dir]:
        for p in root.glob("*.csv"):
            low = p.name.lower()
            if "pred" in low:
                candidates.append(p)

    if not candidates:
        return None

    # Prefer fold-local candidates
    candidates = sorted(candidates, key=lambda x: (x.parent != fold_dir, len(x.name)))
    return candidates[0]


def compute_class_metrics_from_preds(csv_path: Path, thr_binary: float = BINARY_THR) -> dict:
    """
    Computes:
      - Regression MAE (target vs prediction)
      - Binary F1 with threshold thr_binary (<=thr vs >thr)
      - 4-class weighted F1 with CLASS4_CUTS
      - Confusion matrices (binary + 4-class)
    """
    dfp = pd.read_csv(csv_path)

    # Optional: filter to test rows if a "set" column exists and contains 'test'
    set_col = next((c for c in dfp.columns if 'set' in c.lower()), None)
    if set_col is not None:
        if dfp[set_col].astype(str).str.lower().isin(['test']).any():
            dfp = dfp[dfp[set_col].astype(str).str.lower() == 'test'].copy()

    for c in ["patient_ID", "target", "prediction"]:
        if c not in dfp.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")

    # Stable sorting for deterministic plots / tables
    try:
        dfp["patient_ID_num"] = pd.to_numeric(dfp["patient_ID"], errors="coerce")
        dfp = dfp.sort_values(["patient_ID_num", "patient_ID"]).drop(columns=["patient_ID_num"])
    except Exception:
        dfp = dfp.sort_values("patient_ID")

    y_true = dfp["target"].astype(float).values
    y_pred = dfp["prediction"].astype(float).values

    def to_binary(x):
        return 0 if x <= thr_binary else 1

    c1, c2, c3 = CLASS4_CUTS
    def to_class4(x):
        if x < c1:
            return 0
        elif x < c2:
            return 1
        elif x < c3:
            return 2
        else:
            return 3

    mae = float(mean_absolute_error(y_true, y_pred))

    bin_gt   = np.array([to_binary(x) for x in y_true], dtype=int)
    bin_pred = np.array([to_binary(x) for x in y_pred], dtype=int)
    cls_gt   = np.array([to_class4(x) for x in y_true], dtype=int)
    cls_pred = np.array([to_class4(x) for x in y_pred], dtype=int)

    f1_bin = float(f1_score(bin_gt, bin_pred, zero_division=0))
    f1_w4  = float(f1_score(cls_gt, cls_pred, average="weighted", zero_division=0))

    cm_bin = confusion_matrix(bin_gt, bin_pred, labels=[0, 1])
    cm_cls = confusion_matrix(cls_gt, cls_pred, labels=[0, 1, 2, 3])

    return {
        "preds_csv": str(csv_path),
        "mae_from_preds": mae,
        "f1_binary": f1_bin,
        "f1_weighted_4class": f1_w4,
        "cm_binary": cm_bin,
        "cm_4class": cm_cls,
        "n_samples": int(len(dfp)),
    }


def load_runs(runs_dir: Path):
    """Walk runs_dir and collect per-fold results."""
    records = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        agg, img_size, n_frames = parse_run_dir(run_dir)
        if img_size is None:
            continue

        # Try to read config.json for authoritative values
        cfg_path = run_dir / "config.json"
        window_size = None
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            agg         = cfg.get("aggregator", agg)
            img_size    = cfg.get("img_size", img_size)
            n_frames    = cfg.get("n_frames", n_frames)
            window_size = cfg.get("window_size", 0)

        setup = f"s{img_size}_f{n_frames}"

        # Collect fold sub-dirs
        fold_dirs = sorted([d for d in run_dir.iterdir()
                            if d.is_dir() and d.name.startswith("fold")])

        # Also handle flat structure (no sub-folders)
        if not fold_dirs:
            fold_dirs = [run_dir]

        for fold_dir in fold_dirs:
            fold_name = fold_dir.name

            # Read train_log.csv
            csv = fold_dir / "train_log.csv"
            if not csv.exists():
                csv = run_dir / "train_log.csv"
            if not csv.exists():
                continue

            df = pd.read_csv(csv)
            if "val_mae" not in df.columns:
                continue

            df = df.dropna(subset=["val_mae"])
            if df.empty:
                continue

            best_val_mae = float(df["val_mae"].min())
            best_epoch   = int(df.loc[df["val_mae"].idxmin(), "epoch"]) if "epoch" in df.columns else int(df["val_mae"].idxmin() + 1)
            final_tr_loss = float(df["train_loss"].iloc[-1]) if "train_loss" in df.columns else None

            # Optional classification metrics from predictions CSV
            preds_csv = find_preds_csv(fold_dir, run_dir)
            cls_metrics = None
            if preds_csv is not None:
                try:
                    cls_metrics = compute_class_metrics_from_preds(preds_csv, thr_binary=BINARY_THR)
                except Exception as e:
                    cls_metrics = {"error": str(e), "preds_csv": str(preds_csv)}

            records.append(dict(
                run            = run_dir.name,
                aggregator     = agg,
                img_size       = img_size,
                n_frames       = n_frames,
                window_size    = window_size,
                setup          = setup,
                fold           = fold_name,
                best_val_mae   = best_val_mae,
                best_epoch     = best_epoch,
                final_tr_loss  = final_tr_loss,
                n_epochs       = int(len(df)),
                preds_csv      = cls_metrics.get("preds_csv") if isinstance(cls_metrics, dict) else None,
                mae_from_preds = cls_metrics.get("mae_from_preds") if isinstance(cls_metrics, dict) else None,
                f1_binary      = cls_metrics.get("f1_binary") if isinstance(cls_metrics, dict) else None,
                f1_weighted4   = cls_metrics.get("f1_weighted_4class") if isinstance(cls_metrics, dict) else None,
                n_pred_rows    = cls_metrics.get("n_samples") if isinstance(cls_metrics, dict) else None,
                cm_binary      = cls_metrics.get("cm_binary") if isinstance(cls_metrics, dict) else None,
                cm_4class      = cls_metrics.get("cm_4class") if isinstance(cls_metrics, dict) else None,
                log_df         = df,  # keep for learning curves
            ))

    df_out = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["log_df", "cm_binary", "cm_4class"]}
        for r in records
    ])
    return df_out, records


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ± CI95 per (aggregator, setup) for:
      - best_val_mae
      - optional f1_binary
      - optional f1_weighted4
    """
    rows = []
    for (agg, setup), grp in df.groupby(["aggregator", "setup"]):
        vals_mae = grp["best_val_mae"].dropna().values
        row = dict(
            aggregator = agg,
            setup      = setup,
            n_folds    = int(len(vals_mae)),
            mean_mae   = float(np.mean(vals_mae)),
            std_mae    = float(np.std(vals_mae)),
            ci95_mae   = float(ci95(vals_mae)),
            min_mae    = float(np.min(vals_mae)),
            max_mae    = float(np.max(vals_mae)),
        )

        if "f1_binary" in grp.columns and grp["f1_binary"].notna().any():
            vb = grp["f1_binary"].dropna().values
            row.update(dict(
                mean_f1_binary = float(np.mean(vb)),
                std_f1_binary  = float(np.std(vb)),
                ci95_f1_binary = float(ci95(vb)),
                min_f1_binary  = float(np.min(vb)),
                max_f1_binary  = float(np.max(vb)),
            ))

        if "f1_weighted4" in grp.columns and grp["f1_weighted4"].notna().any():
            v4 = grp["f1_weighted4"].dropna().values
            row.update(dict(
                mean_f1_w4 = float(np.mean(v4)),
                std_f1_w4  = float(np.std(v4)),
                ci95_f1_w4 = float(ci95(v4)),
                min_f1_w4  = float(np.min(v4)),
                max_f1_w4  = float(np.max(v4)),
            ))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["aggregator", "setup"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_bar_ci_generic(
    summary: pd.DataFrame,
    out_dir: Path,
    metric_col: str,
    ci_col: str,
    title: str,
    ylabel: str,
    fname: str,
):
    """Generic bar chart: aggregator × setup with CI error bars."""
    if metric_col not in summary.columns:
        print(f"  → [SKIP] {fname} (missing column: {metric_col})")
        return

    aggs   = sorted(summary["aggregator"].unique())
    setups = sorted(summary["setup"].unique())

    n_agg   = len(aggs)
    n_setup = len(setups)
    x       = np.arange(n_agg)
    width   = 0.8 / max(n_setup, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_agg * 2), 6))

    for i, setup in enumerate(setups):
        sub = summary[summary["setup"] == setup].set_index("aggregator")
        means = [sub.loc[a, metric_col] if a in sub.index and pd.notna(sub.loc[a, metric_col]) else np.nan for a in aggs]
        cis   = [sub.loc[a, ci_col]     if a in sub.index and ci_col in sub.columns and pd.notna(sub.loc[a, ci_col]) else 0.0 for a in aggs]

        offset = (i - n_setup / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, means, width * 0.9,
            label=setup, color=PALETTE[i % len(PALETTE)],
            alpha=0.85
        )
        ax.errorbar(
            x + offset, means, yerr=cis, fmt="none",
            color="black", capsize=4, linewidth=1.5
        )

        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.01 if "F1" in ylabel else 0.05),
                    f"{m:.2f}", ha="center", va="bottom", fontsize=8
                )

    ax.set_xticks(x)
    ax.set_xticklabels(aggs, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Setup (size_frames)", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  → {fname}")


def plot_bar_ci_mae(summary: pd.DataFrame, out_dir: Path):
    plot_bar_ci_generic(
        summary=summary,
        out_dir=out_dir,
        metric_col="mean_mae",
        ci_col="ci95_mae",
        title="Aggregator Ablation — Mean Best Val MAE ± 95% CI",
        ylabel="Best Val MAE (lower is better)",
        fname="1_bar_ci_mae.png",
    )


def plot_box(df: pd.DataFrame, out_dir: Path):
    """Box plot per aggregator, colored by setup."""
    setups = sorted(df["setup"].unique())
    aggs   = sorted(df["aggregator"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(aggs) * 2.5), 6))

    positions = []
    data_list = []
    colors    = []
    tick_pos  = []
    tick_lbl  = []

    pos = 0.0
    for agg in aggs:
        group_positions = []
        for i, setup in enumerate(setups):
            sub = df[(df["aggregator"] == agg) & (df["setup"] == setup)]["best_val_mae"].values
            if len(sub) == 0:
                continue
            data_list.append(sub)
            colors.append(PALETTE[i % len(PALETTE)])
            positions.append(pos)
            group_positions.append(pos)
            pos += 1.0
        if group_positions:
            tick_pos.append(float(np.mean(group_positions)))
            tick_lbl.append(agg)
        pos += 0.5  # gap between aggregators

    bp = ax.boxplot(
        data_list, positions=positions, patch_artist=True,
        widths=0.6, showfliers=True
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=30, ha="right")
    ax.set_ylabel("Best Val MAE (lower is better)")
    ax.set_title("Val MAE Distribution per Aggregator")

    legend_patches = [
        mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=s, alpha=0.7)
        for i, s in enumerate(setups)
    ]
    ax.legend(handles=legend_patches, title="Setup", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "2_boxplot.png", dpi=150)
    plt.close(fig)
    print("  → 2_boxplot.png")


def plot_heatmap(summary: pd.DataFrame, out_dir: Path):
    """Heatmap: aggregator × setup → mean val MAE."""
    aggs   = sorted(summary["aggregator"].unique())
    setups = sorted(summary["setup"].unique())

    mat = np.full((len(aggs), len(setups)), np.nan)
    for i, agg in enumerate(aggs):
        for j, setup in enumerate(setups):
            sub = summary[(summary["aggregator"] == agg) & (summary["setup"] == setup)]
            if not sub.empty:
                mat[i, j] = sub["mean_mae"].values[0]

    fig, ax = plt.subplots(figsize=(max(6, len(setups) * 2), max(4, len(aggs) * 0.7)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="Mean Best Val MAE")

    ax.set_xticks(range(len(setups)))
    ax.set_xticklabels(setups, rotation=30, ha="right")
    ax.set_yticks(range(len(aggs)))
    ax.set_yticklabels(aggs)

    median_val = np.nanmedian(mat)
    for i in range(len(aggs)):
        for j in range(len(setups)):
            if not np.isnan(mat[i, j]):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > median_val else "black"
                )

    ax.set_title("Mean Best Val MAE Heatmap (lower is better)")
    ax.set_xlabel("Setup (img_size_n_frames)")
    fig.tight_layout()
    fig.savefig(out_dir / "3_heatmap.png", dpi=150)
    plt.close(fig)
    print("  → 3_heatmap.png")


def plot_learning_curves(records: list, out_dir: Path):
    """Learning curves per aggregator — all folds, all setups."""
    agg_map = {}
    for r in records:
        agg_map.setdefault(r["aggregator"], []).append(r)

    for agg, recs in sorted(agg_map.items()):
        setups = sorted(set(r["setup"] for r in recs))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Learning Curves — {agg}", fontsize=13)

        for i, setup in enumerate(setups):
            color = PALETTE[i % len(PALETTE)]
            setup_recs = [r for r in recs if r["setup"] == setup]

            for r in setup_recs:
                df = r["log_df"]
                if "epoch" in df.columns:
                    x = df["epoch"].values
                else:
                    x = np.arange(1, len(df) + 1)

                if "train_loss" in df.columns:
                    axes[0].plot(x, df["train_loss"], color=color, alpha=0.5, linewidth=1)
                axes[1].plot(x, df["val_mae"], color=color, alpha=0.5, linewidth=1)

            # mean curve
            all_epochs = max(len(r["log_df"]) for r in setup_recs)
            tr_matrix  = np.full((len(setup_recs), all_epochs), np.nan)
            vl_matrix  = np.full((len(setup_recs), all_epochs), np.nan)
            for ri, r in enumerate(setup_recs):
                dff = r["log_df"]
                n  = len(dff)
                if "train_loss" in dff.columns:
                    tr_matrix[ri, :n] = dff["train_loss"].values
                vl_matrix[ri, :n] = dff["val_mae"].values

            mean_tr = np.nanmean(tr_matrix, axis=0)
            mean_vl = np.nanmean(vl_matrix, axis=0)
            ep      = np.arange(1, all_epochs + 1)

            if np.isfinite(mean_tr).any():
                axes[0].plot(ep, mean_tr, color=color, linewidth=2.5, label=setup)
            axes[1].plot(ep, mean_vl, color=color, linewidth=2.5, label=setup)

        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss")
        axes[0].set_title("Train Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val MAE")
        axes[1].set_title("Val MAE"); axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fname = f"4_curves_{agg}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  → {fname}")


def plot_overfitting(df: pd.DataFrame, out_dir: Path):
    """Scatter: final train loss vs best val MAE — bubble = epoch of best val MAE."""
    setups = sorted(df["setup"].unique())

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, setup in enumerate(setups):
        sub = df[df["setup"] == setup].copy()
        sub = sub[pd.notna(sub["final_tr_loss"])]

        if sub.empty:
            continue

        ax.scatter(
            sub["final_tr_loss"], sub["best_val_mae"],
            s=sub["best_epoch"] * 3 + 30,
            c=PALETTE[i % len(PALETTE)],
            alpha=0.7, label=setup, edgecolors="white", linewidths=0.5
        )

    ax.set_xlabel("Final Train Loss")
    ax.set_ylabel("Best Val MAE")
    ax.set_title("Overfitting Analysis (bubble size = epoch of best val MAE)")
    ax.legend(title="Setup")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "5_overfitting.png", dpi=150)
    plt.close(fig)
    print("  → 5_overfitting.png")


def plot_setup_comparison_generic(
    summary: pd.DataFrame,
    out_dir: Path,
    metric_col: str,
    ci_col: str,
    title: str,
    ylabel: str,
    fname: str,
):
    """Line plot: for each aggregator, mean metric across setups with CI band."""
    if metric_col not in summary.columns:
        print(f"  → [SKIP] {fname} (missing column: {metric_col})")
        return

    aggs   = sorted(summary["aggregator"].unique())
    setups = sorted(summary["setup"].unique())

    fig, ax = plt.subplots(figsize=(max(8, len(setups) * 2), 6))

    x = np.arange(len(setups))
    for i, agg in enumerate(aggs):
        sub = summary[summary["aggregator"] == agg].set_index("setup")

        means = np.array([sub.loc[s, metric_col] if s in sub.index and pd.notna(sub.loc[s, metric_col]) else np.nan for s in setups], dtype=float)
        cis   = np.array([sub.loc[s, ci_col]     if s in sub.index and ci_col in sub.columns and pd.notna(sub.loc[s, ci_col]) else 0.0 for s in setups], dtype=float)

        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, means, marker="o", color=color, label=agg, linewidth=2)
        ax.fill_between(x, means - cis, means + cis, color=color, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(setups, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Aggregator", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname}")


def plot_setup_comparison_mae(summary: pd.DataFrame, out_dir: Path):
    plot_setup_comparison_generic(
        summary=summary,
        out_dir=out_dir,
        metric_col="mean_mae",
        ci_col="ci95_mae",
        title="Setup Effect per Aggregator — Mean Best Val MAE ± 95% CI",
        ylabel="Mean Best Val MAE (lower is better)",
        fname="6_setup_comparison_mae.png",
    )


def plot_confmat_grids(records: list, out_dir: Path, which: str = "binary"):
    """
    Confusion matrix grids saved per setup:
      - rows: aggregators
      - cols: folds
    which: "binary" or "4class"
    """
    assert which in ["binary", "4class"]

    setups = sorted(set(r["setup"] for r in records))
    for setup in setups:
        recs = [r for r in records if r["setup"] == setup]

        # only keep those with matrices
        if which == "binary":
            recs = [r for r in recs if r.get("cm_binary") is not None]
        else:
            recs = [r for r in recs if r.get("cm_4class") is not None]

        if not recs:
            continue

        aggs  = sorted(set(r["aggregator"] for r in recs))
        folds = sorted(set(r["fold"] for r in recs))

        nrows = len(aggs)
        ncols = len(folds)

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))

        # normalize axes indexing
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        for i, agg in enumerate(aggs):
            for j, fold in enumerate(folds):
                ax = axes[i, j]
                r = next((x for x in recs if x["aggregator"] == agg and x["fold"] == fold), None)

                if r is None:
                    ax.axis("off")
                    continue

                cm = r["cm_binary"] if which == "binary" else r["cm_4class"]
                if cm is None:
                    ax.axis("off")
                    continue

                ax.imshow(cm, aspect="auto")  # default colormap
                ax.set_xticks([])
                ax.set_yticks([])

                # annotate counts
                for (yy, xx), val in np.ndenumerate(cm):
                    ax.text(xx, yy, str(int(val)), ha="center", va="center", fontsize=9)

                # Titles
                if i == 0:
                    ax.set_title(f"{fold}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(agg, fontsize=10, rotation=0, labelpad=40, va="center")

        fig.suptitle(f"Confusion Matrices ({which}) — {setup}  |  Binary thr={BINARY_THR:g}", fontsize=14, y=0.995)
        fig.tight_layout()

        fname = f"7_confmat_grid_{which}_{setup}.png" if which == "binary" else f"8_confmat_grid_{which}_{setup}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  → {fname}")


def plot_bar_ci_f1(summary: pd.DataFrame, out_dir: Path):
    # Binary F1
    plot_bar_ci_generic(
        summary=summary,
        out_dir=out_dir,
        metric_col="mean_f1_binary",
        ci_col="ci95_f1_binary",
        title=f"Aggregator Ablation — Mean Binary F1 ± 95% CI (thr={BINARY_THR:g})",
        ylabel="Binary F1 (higher is better)",
        fname="9_bar_ci_f1_binary.png",
    )

    # 4-class weighted F1
    plot_bar_ci_generic(
        summary=summary,
        out_dir=out_dir,
        metric_col="mean_f1_w4",
        ci_col="ci95_f1_w4",
        title="Aggregator Ablation — Mean 4-Class Weighted F1 ± 95% CI",
        ylabel="4-Class Weighted F1 (higher is better)",
        fname="10_bar_ci_f1_w4.png",
    )


def plot_setup_comparison_f1(summary: pd.DataFrame, out_dir: Path):
    plot_setup_comparison_generic(
        summary=summary,
        out_dir=out_dir,
        metric_col="mean_f1_binary",
        ci_col="ci95_f1_binary",
        title=f"Setup Effect per Aggregator — Mean Binary F1 ± 95% CI (thr={BINARY_THR:g})",
        ylabel="Mean Binary F1 (higher is better)",
        fname="11_setup_comparison_f1_binary.png",
    )

    plot_setup_comparison_generic(
        summary=summary,
        out_dir=out_dir,
        metric_col="mean_f1_w4",
        ci_col="ci95_f1_w4",
        title="Setup Effect per Aggregator — Mean 4-Class Weighted F1 ± 95% CI",
        ylabel="Mean 4-Class Weighted F1 (higher is better)",
        fname="12_setup_comparison_f1_w4.png",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--out_dir",  type=str, default=str(OUT_DIR))
    parser.add_argument("--min_folds", type=int, default=1,
                        help="Skip (aggregator, setup) groups with fewer than this many completed folds")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading runs from: {runs_dir}")
    df, records = load_runs(runs_dir)

    if df.empty:
        print("No completed runs found. Check --runs_dir.")
        return

    # Filter runs with enough folds for MAE records
    fold_counts = df.groupby(["aggregator", "setup"])["fold"].count()
    valid       = fold_counts[fold_counts >= args.min_folds].index
    df = df[df.set_index(["aggregator", "setup"]).index.isin(valid)].reset_index(drop=True)
    records = [r for r in records if (r["aggregator"], r["setup"]) in valid]

    print(f"Found {len(df)} fold records across "
          f"{df['aggregator'].nunique()} aggregators, "
          f"{df['setup'].nunique()} setups\n")

    summary = summary_table(df)

    # Save summary CSV
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary table saved → {summary_path}\n")
    print(summary.to_string(index=False))
    print()

    print("Generating plots...")
    plot_bar_ci_mae(summary, out_dir)
    plot_box(df, out_dir)
    plot_heatmap(summary, out_dir)
    plot_learning_curves(records, out_dir)
    plot_overfitting(df, out_dir)
    plot_setup_comparison_mae(summary, out_dir)

    # Confusion matrix grids (only if preds CSV exists for at least some folds)
    plot_confmat_grids(records, out_dir, which="binary")
    plot_confmat_grids(records, out_dir, which="4class")

    # F1 plots (only if columns exist in summary)
    plot_bar_ci_f1(summary, out_dir)
    plot_setup_comparison_f1(summary, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()