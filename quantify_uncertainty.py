#!/usr/bin/env python3
"""
quantify_uncertainty.py — Model uncertainty quantification for EUS liver PDFF.

Three complementary methods, each targeting a different source of uncertainty:

  1. MC Stochastic Depth (MCSD)
     Keeps the Swin backbone in train() mode so DropPath (stochastic depth,
     p=0–0.10 per block) remains active during inference. Runs N stochastic
     forward passes and reports mean ± std per video.
     → Captures epistemic uncertainty in the backbone feature extraction.
     → Works for ALL aggregator types (DropPath is in Swin, not the aggregator).

  2. Test-Time Augmentation (TTA)
     Runs inference on T augmented versions of the same bag (random rotation,
     horizontal flip, Gaussian blur). Reports mean ± std across augmentations.
     → Captures input-space uncertainty / sensitivity to image perturbations.
     → Fully model-agnostic; no architecture changes required.

  3. Conformal Prediction (CP)
     Calibrates on the validation set residuals to compute a prediction
     interval [pred − q, pred + q] with a guaranteed (1−α) coverage rate.
     Requires no model modification and provides a mathematically rigorous
     interval with finite-sample validity.
     → The only method that gives a coverage guarantee.

Output (saved to {ckpt_dir}/uncertainty/):
  uncertainty_metrics.csv        — per-video MCSD std, TTA std, CP interval
  calibration_residuals.csv      — val-set residuals used for CP calibration
  fig_uncertainty_overview.png   — scatter: prediction vs GT coloured by uncertainty
  fig_mcsd_distribution.png      — MCSD std distribution across videos
  fig_tta_distribution.png       — TTA std distribution
  fig_conformal_coverage.png     — empirical vs nominal coverage across alpha levels
  fig_case_examples.png          — best/worst cases with uncertainty bars

Usage:
  python quantify_uncertainty.py --ckpt runs_without_test/mh_gated4_s384_f175/fold00/best.pt
  python quantify_uncertainty.py --ckpt ... --n_mc 30 --n_tta 20 --alpha 0.1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

# ── Source path ───────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import set_seed
from data import _load_frames_for_row
from model import MILModel

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


# ─────────────────────────────────────────────────────────────────────────────
# Config (reused from visualize_attention.py)
# ─────────────────────────────────────────────────────────────────────────────

class InferenceCFG:
    def __init__(self, config_json: Path, device: str):
        with open(config_json) as f:
            d = json.load(f)
        self.backbone       = d["backbone"]
        self.aggregator     = d["aggregator"]
        self.img_size       = int(d["img_size"])
        self.n_frames       = int(d["n_frames"])
        self.window_size    = int(d.get("window_size", 0))
        self.video_id_col   = d.get("video_id_col",   "video")
        self.video_path_col = d.get("video_path_col", "video_path")
        self.target_col     = d.get("target_col",     "PDFF")
        self.start_col      = d.get("start_col",      "start_time_1")
        self.end_col        = d.get("end_col",        "end_time_1")
        self.csv_path       = Path(d["csv_path"])
        self.cache_dir      = Path(
            f"/research/projects/Sahika/projects/liver_PDFF/data"
            f"/frame_cache_{int(d['img_size'])}"
        )
        self.set_col         = d.get("set_col", "set1")
        self.n_folds         = int(d.get("n_folds", 0))
        self.fold_set_prefix = d.get("fold_set_prefix", "set")
        self.seed            = int(d.get("seed", 42))
        self.device          = device


def find_config_json(ckpt_path: Path) -> Path:
    for p in [ckpt_path.parent / "config.json",
               ckpt_path.parent.parent / "config.json"]:
        if p.exists():
            return p
    raise FileNotFoundError(f"config.json not found near {ckpt_path}")


def resolve_set_col(cfg: InferenceCFG, fold_dir: Path) -> str:
    if cfg.n_folds <= 0:
        return cfg.set_col
    digits = "".join(c for c in fold_dir.name if c.isdigit())
    if not digits:
        return cfg.set_col
    return f"{cfg.fold_set_prefix}{int(digits) + 1}"


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_bag(
    row: pd.Series,
    cfg: InferenceCFG,
    transform: T.Compose,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load a fixed (linspace-sampled) bag for a video row.
    Returns (bag_tensor: (T,1,H,W), raw_frames: (T,H,W)).
    """
    all_frames  = _load_frames_for_row(row, cfg)
    idxs        = np.linspace(0, all_frames.shape[0] - 1,
                               num=cfg.n_frames, dtype=int)
    frames      = all_frames[idxs]
    bag = torch.stack([
        transform(
            Image.fromarray(fr, mode="L") if fr.ndim == 2
            else Image.fromarray(fr).convert("L")
        )
        for fr in frames
    ])
    return bag.to(cfg.device), frames


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — MC Stochastic Depth
# ─────────────────────────────────────────────────────────────────────────────

def enable_mc_stochastic_depth(model: MILModel) -> None:
    """
    Set the Swin backbone to train() mode so DropPath layers fire stochastically,
    while keeping BatchNorm and the aggregator/regressor in eval() mode.

    DropPath (stochastic depth) in the pretrained Swin Tiny has probabilities
    0.009–0.100 linearly increasing from first to last block. These are only
    active in training mode.  The attn_drop and proj_drop have p=0 in this
    checkpoint so they contribute no stochasticity.
    """
    model.eval()                        # freeze BN, aggregator, regressor
    model.backbone.train()              # activate DropPath in backbone
    # Keep BatchNorm frozen inside backbone
    for m in model.backbone.modules():
        if isinstance(m, (torch.nn.BatchNorm1d,
                          torch.nn.BatchNorm2d,
                          torch.nn.LayerNorm)):
            m.eval()


def mc_stochastic_depth(
    model: MILModel,
    bag: torch.Tensor,
    n_passes: int = 30,
) -> tuple[float, float, np.ndarray]:
    """
    Run n_passes stochastic forward passes with DropPath active in Swin.

    Returns
    ───────
    mean_pred  : float
    std_pred   : float
    all_preds  : (n_passes,) array
    """
    enable_mc_stochastic_depth(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(model(bag).item())
    model.eval()  # restore full eval
    arr = np.array(preds)
    return float(arr.mean()), float(arr.std()), arr


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — Test-Time Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def build_tta_transform(img_size: int, aug_idx: int) -> T.Compose:
    """
    Build a deterministic augmentation transform from an index.

    We enumerate a small fixed grid of augmentations so results are
    reproducible without needing to fix random seeds per pass:
      - Identity
      - Horizontal flip
      - Rotation ±10°, ±20°
      - Gaussian blur σ=0.5, 1.0
      - Flip + rotation combinations

    Falls back to random augmentation for aug_idx >= num_presets.
    """
    base = [T.Resize((img_size, img_size)), T.ToTensor()]

    presets = [
        [],                                            # 0: identity
        [T.RandomHorizontalFlip(p=1.0)],               # 1: hflip
        [T.RandomRotation(degrees=(10, 10))],          # 2: rot +10
        [T.RandomRotation(degrees=(-10, -10))],        # 3: rot -10
        [T.RandomRotation(degrees=(20, 20))],          # 4: rot +20
        [T.RandomRotation(degrees=(-20, -20))],        # 5: rot -20
        [T.GaussianBlur(3, sigma=0.5)],                # 6: blur light
        [T.GaussianBlur(3, sigma=1.5)],                # 7: blur heavy
        [T.RandomHorizontalFlip(p=1.0),
         T.RandomRotation(degrees=(10, 10))],          # 8: hflip + rot
        [T.RandomHorizontalFlip(p=1.0),
         T.GaussianBlur(3, sigma=0.5)],                # 9: hflip + blur
    ]

    if aug_idx < len(presets):
        ops = presets[aug_idx]
    else:
        # Random augmentation for aug_idx >= len(presets)
        ops = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.GaussianBlur(3, sigma=(0.1, 1.5)),
        ]
    return T.Compose(ops + base)


def test_time_augmentation(
    model: MILModel,
    raw_frames: np.ndarray,
    cfg: InferenceCFG,
    n_augmentations: int = 20,
) -> tuple[float, float, np.ndarray]:
    """
    Run inference on n_augmentations differently-augmented versions of the
    same raw frames.  Model stays in eval() throughout.

    Returns
    ───────
    mean_pred      : float
    std_pred       : float
    all_preds      : (n_augmentations,) array
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(n_augmentations):
            transform = build_tta_transform(cfg.img_size, i)
            bag = torch.stack([
                transform(
                    Image.fromarray(fr, mode="L") if fr.ndim == 2
                    else Image.fromarray(fr).convert("L")
                )
                for fr in raw_frames
            ]).to(cfg.device)
            preds.append(model(bag).item())

    arr = np.array(preds)
    return float(arr.mean()), float(arr.std()), arr


# ─────────────────────────────────────────────────────────────────────────────
# Method 3 — Conformal Prediction
# ─────────────────────────────────────────────────────────────────────────────

def compute_conformal_calibration(
    val_preds: np.ndarray,
    val_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute conformity scores (absolute residuals) on the calibration set.

    For regression conformal prediction we use the absolute residual as
    the non-conformity score:  s_i = |y_i - ŷ_i|

    Returns the sorted residuals array for quantile lookup.
    """
    residuals = np.abs(val_preds - val_labels)
    return np.sort(residuals)


def conformal_quantile(
    calibration_residuals: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Compute the (1−α) quantile of calibration residuals with the
    Venn–Conformal finite-sample correction:
      q = ceil((n+1)(1−α)) / n  quantile of sorted residuals.

    This guarantees marginal coverage:
      P(y_new ∈ [ŷ − q, ŷ + q]) ≥ 1 − α

    Args:
        calibration_residuals : sorted absolute residuals from val set
        alpha                 : miscoverage level (0.1 → 90% coverage)
    Returns:
        q : half-width of the prediction interval
    """
    n   = len(calibration_residuals)
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    idx = min(idx, n - 1)  # clip to valid range (when alpha very small)
    return float(calibration_residuals[idx])


def empirical_coverage(
    val_preds: np.ndarray,
    val_labels: np.ndarray,
    calibration_residuals: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """
    Compute empirical coverage at each nominal level for the calibration plot.
    Uses leave-one-out: for each point i, calibrate on all others, predict i.
    """
    n = len(val_preds)
    coverages = np.zeros(len(alphas))

    for i, alpha in enumerate(alphas):
        covered = 0
        for j in range(n):
            # LOO calibration: exclude point j
            loo_resid = np.delete(np.abs(val_preds - val_labels), j)
            loo_resid = np.sort(loo_resid)
            q = conformal_quantile(loo_resid, alpha)
            if abs(val_preds[j] - val_labels[j]) <= q:
                covered += 1
        coverages[i] = covered / n

    return coverages


# ─────────────────────────────────────────────────────────────────────────────
# Full val-set inference (deterministic, eval mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_val_inference(
    model: MILModel,
    val_df: pd.DataFrame,
    cfg: InferenceCFG,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Deterministic inference on the full validation set.
    Returns (preds, labels, video_ids).
    """
    transform = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
    ])
    model.eval()
    preds, labels, vids = [], [], []

    for i in range(len(val_df)):
        row = val_df.iloc[i]
        vid = str(row[cfg.video_id_col])
        gt  = float(row[cfg.target_col])
        try:
            bag, _ = load_bag(row, cfg, transform)
            with torch.no_grad():
                pred = model(bag).item()
            preds.append(pred)
            labels.append(gt)
            vids.append(vid)
            print(f"  [{i+1}/{len(val_df)}] {vid}: GT={gt:.2f}  Pred={pred:.2f}")
        except Exception as exc:
            print(f"  [{i+1}/{len(val_df)}] {vid}: ERROR — {exc}")

    return np.array(preds), np.array(labels), vids


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_overview(
    df: pd.DataFrame,
    cp_q: float,
    alpha: float,
    out_dir: Path,
) -> None:
    """
    4-panel overview:
      (a) Pred vs GT with MCSD uncertainty bars
      (b) Pred vs GT with TTA uncertainty bars
      (c) Pred vs GT with CP interval (uniform band)
      (d) MCSD std vs TTA std coloured by MAE
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    lim = max(df["gt_pdff"].max(), df["pred_pdff"].max()) + 3

    for ax, unc_col, title, color in [
        (axes[0, 0], "mcsd_std", f"MC Stochastic Depth (σ)", "#1565C0"),
        (axes[0, 1], "tta_std",  f"Test-Time Augmentation (σ)", "#2E7D32"),
    ]:
        eb = ax.errorbar(
            df["gt_pdff"], df["pred_pdff"],
            yerr=df[unc_col],
            fmt="o", color=color, ecolor=color, elinewidth=0.8,
            capsize=3, markersize=4, alpha=0.75, label="±1σ",
        )
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("GT PDFF (%)")
        ax.set_ylabel("Predicted PDFF (%)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(-1, lim)
        ax.set_ylim(-1, lim)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    # CP panel — uniform band
    ax = axes[1, 0]
    ax.scatter(df["gt_pdff"], df["pred_pdff"],
               c=df["mae"], cmap="RdYlGn_r", s=40,
               edgecolors="black", linewidth=0.3, zorder=3)
    ax.fill_between([0, lim],
                    [0 - cp_q, lim - cp_q],
                    [0 + cp_q, lim + cp_q],
                    alpha=0.15, color="#F57F17",
                    label=f"CP interval (±{cp_q:.2f}%)")
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.4)
    covered = (np.abs(df["pred_pdff"] - df["gt_pdff"]) <= cp_q).mean()
    ax.set_xlabel("GT PDFF (%)")
    ax.set_ylabel("Predicted PDFF (%)")
    ax.set_title(
        f"Conformal Prediction  [{100*(1-alpha):.0f}% nominal]\n"
        f"Empirical coverage: {covered:.1%}  |  Half-width: {cp_q:.2f}%",
        fontweight="bold",
    )
    ax.set_xlim(-1, lim)
    ax.set_ylim(-1, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    plt.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(df["mae"].min(), df["mae"].max()),
        cmap="RdYlGn_r"), ax=ax, label="MAE")

    # MCSD vs TTA scatter
    ax = axes[1, 1]
    sc = ax.scatter(df["mcsd_std"], df["tta_std"],
                    c=df["mae"], cmap="RdYlGn_r", s=50,
                    edgecolors="black", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label="MAE")
    ax.set_xlabel("MCSD σ  (epistemic, backbone DropPath)")
    ax.set_ylabel("TTA σ  (input sensitivity)")
    ax.set_title("Uncertainty Method Comparison\n(coloured by MAE)",
                 fontweight="bold")
    # Correlation
    r = np.corrcoef(df["mcsd_std"], df["tta_std"])[0, 1]
    ax.text(0.05, 0.93, f"r = {r:.2f}", transform=ax.transAxes, fontsize=9)

    fig.suptitle(
        "Uncertainty Quantification Overview — "
        f"mh_gated4  (n={len(df)} val videos)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, out_dir, "fig_uncertainty_overview")


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Histogram + CDF of MCSD and TTA uncertainty, stratified by PDFF level.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    df["pdff_cat"] = pd.cut(df["gt_pdff"],
                             bins=[0, 5, 10, 100],
                             labels=["<5%", "5–10%", ">10%"])
    colors = {"<5%": "#1565C0", "5–10%": "#F57F17", ">10%": "#C62828"}

    for row_idx, (col, title, xlabel) in enumerate([
        ("mcsd_std", "MC Stochastic Depth", "σ (PDFF %)"),
        ("tta_std",  "Test-Time Augmentation", "σ (PDFF %)"),
    ]):
        # Histogram
        ax = axes[row_idx, 0]
        for cat, grp in df.groupby("pdff_cat", observed=True):
            ax.hist(grp[col], bins=15, alpha=0.6, color=colors[str(cat)],
                    label=f"PDFF {cat} (n={len(grp)})", edgecolor="white",
                    linewidth=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"{title} — Distribution by PDFF Level")
        ax.legend(fontsize=8)
        ax.axvline(df[col].median(), color="black", linestyle="--",
                   linewidth=1, label="Median")

        # CDF
        ax = axes[row_idx, 1]
        for cat, grp in df.groupby("pdff_cat", observed=True):
            sorted_vals = np.sort(grp[col])
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf, color=colors[str(cat)],
                    label=f"PDFF {cat}", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative Fraction")
        ax.set_title(f"{title} — CDF")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    savefig(fig, out_dir, "fig_uncertainty_distributions")


def plot_conformal_coverage(
    val_preds: np.ndarray,
    val_labels: np.ndarray,
    calibration_residuals: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Reliability plot: empirical coverage vs nominal (1−α) level.
    A well-calibrated conformal predictor follows the diagonal.
    Uses the split-conformal (not LOO) approximation for speed.
    """
    alphas = np.linspace(0.05, 0.5, 20)
    nominal = 1 - alphas

    # Split conformal: use all val residuals as calibration
    # (slight optimism since same set used for both — plot is illustrative)
    empirical = []
    for a in alphas:
        q       = conformal_quantile(calibration_residuals, a)
        covered = (np.abs(val_preds - val_labels) <= q).mean()
        empirical.append(covered)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Coverage plot
    ax = axes[0]
    ax.plot(nominal, empirical, "o-", color="#1565C0", linewidth=2,
            markersize=5, label="Empirical coverage")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Ideal")
    ax.fill_between(nominal, nominal, empirical, alpha=0.15,
                    color="#1565C0", label="Coverage gap")
    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal Prediction Reliability\n"
                 "(points above diagonal = conservative)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_aspect("equal")

    # (b) Interval width vs nominal coverage
    widths = [2 * conformal_quantile(calibration_residuals, a) for a in alphas]
    ax = axes[1]
    ax.plot(nominal, widths, "s-", color="#C62828", linewidth=2, markersize=5)
    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Prediction interval width (PDFF %)")
    ax.set_title("Interval Width vs Coverage Level", fontweight="bold")
    ax.grid(alpha=0.3)
    for a_mark in [0.10, 0.20]:
        q_mark = 2 * conformal_quantile(calibration_residuals, a_mark)
        ax.axvline(1 - a_mark, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(q_mark, color="gray", linestyle=":", linewidth=0.8)
        ax.text(1 - a_mark + 0.005, q_mark + 0.2,
                f"{100*(1-a_mark):.0f}%→±{q_mark/2:.2f}%",
                fontsize=8, color="#C62828")

    plt.tight_layout()
    savefig(fig, out_dir, "fig_conformal_coverage")


def plot_case_examples(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Horizontal bar chart showing predictions ± uncertainty for:
      - 5 most confident correct cases  (low MCSD, low MAE)
      - 5 most uncertain cases          (high MCSD)
      - 5 worst-MAE cases
    """
    df = df.copy().sort_values("gt_pdff")
    df["mcsd_cv"] = df["mcsd_std"] / (df["pred_pdff"].abs() + 1e-3)  # coeff of variation

    confident = df.nsmallest(6, "mcsd_std")
    uncertain = df.nlargest(6, "mcsd_std")
    worst     = df.nlargest(6, "mae")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax, subset, title, color in [
        (axes[0], confident, "Most Confident (low MCSD σ)", "#1565C0"),
        (axes[1], uncertain, "Most Uncertain (high MCSD σ)", "#E65100"),
        (axes[2], worst,     "Largest Errors (high MAE)",   "#C62828"),
    ]:
        labels = subset["video"].values
        preds  = subset["pred_pdff"].values
        gts    = subset["gt_pdff"].values
        errs   = subset["mcsd_std"].values

        y = np.arange(len(labels))
        ax.barh(y, preds, xerr=errs, color=color, alpha=0.6, height=0.4,
                capsize=4, label="Pred ± MCSD σ")
        ax.scatter(gts, y, color="black", zorder=5, s=40, marker="D",
                   label="GT PDFF")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("PDFF (%)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.invert_yaxis()

    fig.suptitle("Case Examples with MC Stochastic Depth Uncertainty",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir, "fig_case_examples")


def plot_uncertainty_vs_error(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Key clinical plot: does high uncertainty predict high error?
    If yes → the model 'knows when it doesn't know'.
    """
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, unc_col, label, color in [
        (axes[0], "mcsd_std", "MCSD σ", "#1565C0"),
        (axes[1], "tta_std",  "TTA σ",  "#2E7D32"),
    ]:
        ax.scatter(df[unc_col], df["mae"],
                   c=df["gt_pdff"], cmap="plasma", s=50,
                   edgecolors="black", linewidth=0.4, alpha=0.8)
        # Regression line
        slope, intercept, r, p, _ = stats.linregress(df[unc_col], df["mae"])
        x_line = np.linspace(df[unc_col].min(), df[unc_col].max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color="red", linewidth=1.5, linestyle="--",
                label=f"r={r:.2f}, p={p:.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("MAE (|pred − GT|)")
        ax.set_title(f"Uncertainty vs Prediction Error\n({label})",
                     fontweight="bold")
        ax.legend(fontsize=9)

    plt.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(df["gt_pdff"].min(), df["gt_pdff"].max()),
        cmap="plasma"), ax=axes[1], label="GT PDFF (%)")

    fig.suptitle("Does uncertainty predict error?  (ideal: r > 0, p < 0.05)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir, "fig_uncertainty_vs_error")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",  required=True, type=Path,
                   help="Path to best.pt")
    p.add_argument("--n_mc",  type=int, default=30,
                   help="MC Stochastic Depth passes (default: 30)")
    p.add_argument("--n_tta", type=int, default=20,
                   help="TTA augmentation variants (default: 20)")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Conformal miscoverage level (default: 0.1 → 90%% CI)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    ckpt_path = args.ckpt.resolve()
    fold_dir  = ckpt_path.parent

    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    config_path = find_config_json(ckpt_path)
    cfg         = InferenceCFG(config_path, args.device)
    cfg.set_col = resolve_set_col(cfg, fold_dir)

    out_dir = fold_dir / "uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCheckpoint : {ckpt_path}")
    print(f"Config     : {config_path}")
    print(f"Aggregator : {cfg.aggregator}")
    print(f"Split col  : {cfg.set_col}")
    print(f"MC passes  : {args.n_mc}")
    print(f"TTA passes : {args.n_tta}")
    print(f"CP alpha   : {args.alpha}  ({100*(1-args.alpha):.0f}% CI)")
    print(f"Output     : {out_dir}\n")

    set_seed(cfg.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    model = MILModel(cfg).to(args.device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=args.device, weights_only=True)
    )
    model.eval()
    print(f"Loaded: {type(model.aggregator).__name__}  "
          f"({model.backbone.num_features}d)\n")

    # ── Validation split ──────────────────────────────────────────────────────
    df_all = pd.read_csv(cfg.csv_path)
    val_df = df_all[df_all[cfg.set_col] == "val"].reset_index(drop=True)
    if val_df.empty:
        val_df = df_all[df_all[cfg.set_col] == "train"].reset_index(drop=True)
        print(f"[WARN] No 'val' rows; falling back to 'train' split.")
    print(f"Validation videos: {len(val_df)}\n")

    base_transform = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
    ])

    # ── Step 1: Deterministic val inference (for CP calibration) ─────────────
    print("=" * 60)
    print("Step 1/3 — Deterministic val-set inference (CP calibration)")
    print("=" * 60)
    val_preds, val_labels, val_vids = run_val_inference(model, val_df, cfg)
    calib_residuals = compute_conformal_calibration(val_preds, val_labels)
    cp_q            = conformal_quantile(calib_residuals, args.alpha)

    pd.DataFrame({
        "video": val_vids,
        "gt_pdff": val_labels,
        "pred_pdff": val_preds,
        "residual": np.abs(val_preds - val_labels),
    }).to_csv(out_dir / "calibration_residuals.csv", index=False)
    print(f"\nCP half-width q={cp_q:.3f}% at {100*(1-args.alpha):.0f}% nominal coverage")
    print(f"Empirical coverage: "
          f"{(np.abs(val_preds - val_labels) <= cp_q).mean():.1%}\n")

    # ── Step 2: MCSD + TTA per video ─────────────────────────────────────────
    print("=" * 60)
    print(f"Step 2/3 — MCSD ({args.n_mc} passes) + TTA ({args.n_tta} augmentations)")
    print("=" * 60)

    rows = []
    for i in range(len(val_df)):
        row    = val_df.iloc[i]
        vid_id = str(row[cfg.video_id_col])
        gt     = float(row[cfg.target_col])

        try:
            bag, raw_frames = load_bag(row, cfg, base_transform)

            # MCSD
            mc_mean, mc_std, mc_arr = mc_stochastic_depth(
                model, bag, n_passes=args.n_mc
            )
            # TTA
            tta_mean, tta_std, tta_arr = test_time_augmentation(
                model, raw_frames, cfg, n_augmentations=args.n_tta
            )

            # Deterministic pred (already computed above, reuse)
            det_pred = float(val_preds[i]) if i < len(val_preds) else mc_mean

            rows.append({
                "video":       vid_id,
                "gt_pdff":     gt,
                "pred_pdff":   det_pred,
                "mae":         abs(det_pred - gt),
                "mcsd_mean":   mc_mean,
                "mcsd_std":    mc_std,
                "mcsd_cv":     mc_std / (abs(mc_mean) + 1e-3),
                "tta_mean":    tta_mean,
                "tta_std":     tta_std,
                "cp_lower":    det_pred - cp_q,
                "cp_upper":    det_pred + cp_q,
                "cp_covered":  float(abs(det_pred - gt) <= cp_q),
            })

            print(f"  [{i+1:2d}/{len(val_df)}] {vid_id:<15s}  "
                  f"GT={gt:5.2f}  Pred={det_pred:5.2f}  "
                  f"MCSD_σ={mc_std:.3f}  TTA_σ={tta_std:.3f}")

        except Exception as exc:
            print(f"  [{i+1:2d}/{len(val_df)}] {vid_id}: ERROR — {exc}")

    results_df = pd.DataFrame(rows)
    csv_path   = out_dir / "uncertainty_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV: {csv_path}")

    # ── Step 3: Figures ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3/3 — Generating figures")
    print("=" * 60)

    plot_uncertainty_overview(results_df, cp_q, args.alpha, out_dir)
    plot_distributions(results_df, out_dir)
    plot_conformal_coverage(val_preds, val_labels, calib_residuals, out_dir)
    plot_case_examples(results_df, out_dir)
    plot_uncertainty_vs_error(results_df, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("UNCERTAINTY SUMMARY")
    print("=" * 60)
    print(f"\nMethod 1 — MC Stochastic Depth ({args.n_mc} passes, Swin DropPath active)")
    print(f"  Mean σ   : {results_df['mcsd_std'].mean():.3f} ± "
          f"{results_df['mcsd_std'].std():.3f} PDFF %")
    print(f"  Range σ  : [{results_df['mcsd_std'].min():.3f}, "
          f"{results_df['mcsd_std'].max():.3f}]")
    print(f"  Source   : Swin DropPath (p=0.009–0.100 per block)")

    print(f"\nMethod 2 — Test-Time Augmentation ({args.n_tta} augmentations)")
    print(f"  Mean σ   : {results_df['tta_std'].mean():.3f} ± "
          f"{results_df['tta_std'].std():.3f} PDFF %")
    print(f"  Range σ  : [{results_df['tta_std'].min():.3f}, "
          f"{results_df['tta_std'].max():.3f}]")

    print(f"\nMethod 3 — Conformal Prediction (α={args.alpha})")
    print(f"  Calibration n : {len(calib_residuals)} val residuals")
    print(f"  Interval      : pred ± {cp_q:.3f}% PDFF "
          f"({100*(1-args.alpha):.0f}% nominal coverage)")
    print(f"  Empirical cov : {results_df['cp_covered'].mean():.1%} "
          f"on val set (should be ≥ {100*(1-args.alpha):.0f}%)")

    from scipy import stats
    r_mc,  p_mc  = stats.pearsonr(results_df["mcsd_std"], results_df["mae"])
    r_tta, p_tta = stats.pearsonr(results_df["tta_std"],  results_df["mae"])
    print(f"\nUncertainty–Error correlation")
    print(f"  MCSD σ vs MAE : r={r_mc:.3f}  (p={p_mc:.4f})")
    print(f"  TTA  σ vs MAE : r={r_tta:.3f}  (p={p_tta:.4f})")
    sig = lambda p: "✓ significant" if p < 0.05 else "✗ not significant"
    print(f"  {'→ Model uncertainty is predictive of error' if p_mc < 0.05 else '→ Uncertainty does NOT predict error (overconfident model)'}")

    print(f"\nAll outputs: {out_dir}")


if __name__ == "__main__":
    main()
