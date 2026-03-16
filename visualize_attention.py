#!/usr/bin/env python3
"""
visualize_attention.py — Attention analysis for the EUS liver PDFF MIL model.

Extracts and saves attention visualisations at two levels:
  1. MIL-level  — per-frame attention weights (which frames matter?)
  2. Swin-level — spatial attention heatmaps (which regions matter?)

All configuration is read from the config.json that lives alongside best.pt.

Config search order (first found wins):
  1. {ckpt_dir}/config.json          (fold-level, if it exists)
  2. {ckpt_dir}/../config.json        (run-level, the typical location)

Output is written to:
  {ckpt_dir}/attention_analysis/{video_id}/

Usage:
  # single video (default: first entry in the validation split)
  python visualize_attention.py --ckpt /path/to/best.pt

  # different video index or more top-K frames
  python visualize_attention.py --ckpt /path/to/best.pt --sample_idx 3 --top_k 6

  # all validation videos + cross-video summary
  python visualize_attention.py --ckpt /path/to/best.pt --all_videos
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
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def find_config_json(ckpt_path: Path) -> Path:
    """
    Search for config.json starting from the checkpoint's directory,
    then one level up (the run/model directory).

    Typical layout:
      runs_with_test/mh_gated4_s384_f175/config.json   ← run-level
      runs_with_test/mh_gated4_s384_f175/fold00/best.pt
    """
    fold_dir = ckpt_path.parent
    candidates = [
        fold_dir / "config.json",           # fold-level (uncommon)
        fold_dir.parent / "config.json",    # run-level  (typical)
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"config.json not found near {ckpt_path}.\n"
        f"Searched:\n" + "\n".join(f"  {p}" for p in candidates)
    )


class InferenceCFG:
    """
    Lightweight config built entirely from a saved config.json.
    Does not call __post_init__, create directories, or write files.
    """
    def __init__(self, config_json: Path, device: str):
        with open(config_json) as f:
            d = json.load(f)

        # ── Model ──────────────────────────────────────────────────────────
        self.backbone    = d["backbone"]
        self.aggregator  = d["aggregator"]
        self.img_size    = int(d["img_size"])
        self.n_frames    = int(d["n_frames"])
        self.window_size = int(d.get("window_size", 0))

        # ── Data columns ───────────────────────────────────────────────────
        self.video_id_col   = d.get("video_id_col",   "video")
        self.video_path_col = d.get("video_path_col", "video_path")
        self.target_col     = d.get("target_col",     "PDFF")
        self.start_col      = d.get("start_col",      "start_time_1")
        self.end_col        = d.get("end_col",        "end_time_1")

        # ── Data paths ─────────────────────────────────────────────────────
        self.csv_path  = Path(d["csv_path"])
        self.cache_dir = Path(
            f"/research/projects/Sahika/projects/liver_PDFF/data"
            f"/frame_cache_{self.img_size}"
        )

        # ── Split column ───────────────────────────────────────────────────
        # Stored as a plain attribute; caller may override after construction
        # when n_folds > 0 to use the per-fold column (set1 … set5).
        self.set_col         = d.get("set_col", "set1")
        self.n_folds         = int(d.get("n_folds", 0))
        self.fold_set_prefix = d.get("fold_set_prefix", "set")

        # ── Runtime ────────────────────────────────────────────────────────
        self.seed   = int(d.get("seed", 42))
        self.device = device


def resolve_set_col(cfg: InferenceCFG, fold_dir: Path) -> str:
    """
    When the experiment used k-fold CV (n_folds > 0), each fold has its own
    split column (e.g. set1 … set5).  We infer the fold index from the
    directory name ('fold00' → 0, 'fold01' → 1, …).

    Falls back to cfg.set_col if the name cannot be parsed or n_folds == 0.
    """
    if cfg.n_folds <= 0:
        return cfg.set_col

    name = fold_dir.name  # e.g. "fold00", "fold01"
    digits = "".join(c for c in name if c.isdigit())
    if not digits:
        return cfg.set_col

    fold_idx = int(digits)          # fold00 → 0, fold03 → 3
    return f"{cfg.fold_set_prefix}{fold_idx + 1}"   # → "set1", "set4", …


# ─────────────────────────────────────────────────────────────────────────────
# Figure saving
# ─────────────────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MIL attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_mil_attention(
    model: MILModel, bag_tensor: torch.Tensor
) -> tuple[float, np.ndarray, torch.Tensor]:
    """
    Forward-pass the bag and capture per-frame MIL attention weights.

    Supported aggregators
    ─────────────────────
    ABMILPool / TemporalABMILPool
        Patches the gated-attention forward to capture softmax(A).
    AttentionMIL
        Captures softmax(att(x)).
    GatedAttentionMIL
        Captures softmax(w(sigmoid(U)⊙tanh(V))).
    MultiHeadGatedAttn
        Patches every head independently; returns head-averaged alpha.
    TemporalConvGatedMIL
        Patches the inner MultiHeadGatedAttn pool after the conv.
    MeanPool
        Returns uniform weights (1/T each).
    MaxPool
        Returns feature-L2-norm as a proxy.

    Returns
    ───────
    pred   : scalar PDFF prediction
    alpha  : (T,) per-frame attention weights
    feats  : (T, D) backbone frame features
    """
    aggregator = model.aggregator
    agg_class  = type(aggregator).__name__

    captured         = {"alpha": None}
    original_forward = aggregator.forward
    orig_head_fwds   = []   # populated for multi-head variants

    # ── ABMILPool / TemporalABMILPool ────────────────────────────────────────
    if agg_class in ("ABMILPool", "TemporalABMILPool"):
        def patched_forward(x):
            if agg_class == "TemporalABMILPool":
                x = x + aggregator.tpeg(x)
            h = aggregator.patch_embed(x)
            A = (aggregator.attn_c(aggregator.attn_a(h) * aggregator.attn_b(h))
                 if aggregator.gate else aggregator.attn(h))
            A = torch.softmax(A, dim=0)
            captured["alpha"] = A.detach().cpu().squeeze(-1).numpy()
            return aggregator.proj_back((A * h).sum(0))
        aggregator.forward = patched_forward

    # ── AttentionMIL ──────────────────────────────────────────────────────────
    elif agg_class == "AttentionMIL":
        def patched_forward(x):
            alpha = torch.softmax(aggregator.att(x), 0)
            captured["alpha"] = alpha.detach().cpu().squeeze(-1).numpy()
            return (alpha * x).sum(0)
        aggregator.forward = patched_forward

    # ── GatedAttentionMIL ────────────────────────────────────────────────────
    elif agg_class == "GatedAttentionMIL":
        def patched_forward(x):
            a     = torch.sigmoid(aggregator.U(x)) * torch.tanh(aggregator.V(x))
            alpha = torch.softmax(aggregator.w(a), 0)
            captured["alpha"] = alpha.detach().cpu().squeeze(-1).numpy()
            return (alpha * x).sum(0)
        aggregator.forward = patched_forward

    # ── MultiHeadGatedAttn ───────────────────────────────────────────────────
    elif agg_class == "MultiHeadGatedAttn":
        head_alphas = []

        def _make_head_patch(hmod):
            def patched_head(x):
                a     = torch.sigmoid(hmod.U(x)) * torch.tanh(hmod.V(x))
                alpha = torch.softmax(hmod.w(a), 0)
                head_alphas.append(alpha.detach().cpu().squeeze(-1).numpy())
                return (alpha * x).sum(0)
            return patched_head

        for head in aggregator.heads:
            orig_head_fwds.append(head.forward)
            head.forward = _make_head_patch(head)

        def patched_forward(x):
            head_alphas.clear()
            out = torch.cat([h(x) for h in aggregator.heads], dim=-1)
            captured["alpha"] = np.stack(head_alphas).mean(axis=0)
            return aggregator.proj(out)
        aggregator.forward = patched_forward

    # ── TemporalConvGatedMIL ─────────────────────────────────────────────────
    elif agg_class == "TemporalConvGatedMIL":
        pool        = aggregator.pool
        head_alphas = []

        def _make_head_patch(hmod):
            def patched_head(x):
                a     = torch.sigmoid(hmod.U(x)) * torch.tanh(hmod.V(x))
                alpha = torch.softmax(hmod.w(a), 0)
                head_alphas.append(alpha.detach().cpu().squeeze(-1).numpy())
                return (alpha * x).sum(0)
            return patched_head

        for head in pool.heads:
            orig_head_fwds.append(head.forward)
            head.forward = _make_head_patch(head)

        def patched_forward(x):
            head_alphas.clear()
            feats = aggregator.conv(x.T.unsqueeze(0)).squeeze(0).T
            out   = torch.cat([h(feats) for h in pool.heads], dim=-1)
            captured["alpha"] = np.stack(head_alphas).mean(axis=0)
            return pool.proj(out)
        aggregator.forward = patched_forward

    elif agg_class not in ("MeanPool", "MaxPool"):
        print(f"[WARN] Unsupported aggregator '{agg_class}'. Returning uniform proxy.")

    # ── Forward pass with backbone feature capture ────────────────────────────
    backbone_out = {}
    hook = model.backbone.register_forward_hook(
        lambda m, i, o: backbone_out.update({"feats": o.detach().cpu()})
    )
    with torch.no_grad():
        pred = model(bag_tensor).item()
    hook.remove()

    # ── Restore original forwards ─────────────────────────────────────────────
    aggregator.forward = original_forward
    if agg_class == "MultiHeadGatedAttn":
        for head, orig in zip(aggregator.heads, orig_head_fwds):
            head.forward = orig
    elif agg_class == "TemporalConvGatedMIL":
        for head, orig in zip(aggregator.pool.heads, orig_head_fwds):
            head.forward = orig

    # ── Fallback proxies for non-attention poolers ────────────────────────────
    if captured["alpha"] is None:
        feats = backbone_out["feats"]
        if agg_class == "MaxPool":
            norms = feats.norm(dim=-1).numpy()
            captured["alpha"] = norms / norms.sum()
        else:
            T = bag_tensor.shape[0]
            captured["alpha"] = np.ones(T) / T

    return pred, captured["alpha"], backbone_out["feats"]


# ─────────────────────────────────────────────────────────────────────────────
# Swin attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def _swin_attn_hook(module, inp, target_block) -> torch.Tensor:
    """
    Recompute Q@K^T + bias + optional shift mask → softmax inside a
    WindowAttention forward hook.

    fused_attn is disabled by the caller so the non-fused branch runs and
    we can intercept the (nW*B, N, C) inputs.

    Returns (nW*B, heads, N, N) float32 on CPU.
    """
    x = inp[0]                          # (nW*B, N, C)
    B_, N, C = x.shape
    qkv = (module.qkv(x)
           .reshape(B_, N, 3, module.num_heads, -1)
           .permute(2, 0, 3, 1, 4))
    q, k, _ = qkv.unbind(0)
    attn = (q * module.scale) @ k.transpose(-2, -1) + module._get_rel_pos_bias()

    mask = getattr(target_block, "attn_mask", None)
    if mask is not None and not getattr(target_block, "dynamic_mask", False):
        nW   = mask.shape[0]
        attn = (attn.view(-1, nW, module.num_heads, N, N)
                + mask.unsqueeze(1).unsqueeze(0)).view(-1, module.num_heads, N, N)

    return torch.softmax(attn, dim=-1).detach().cpu()


def _grid_size_at_stage(img_size: int, n_stages: int, stage: int) -> int:
    """Patch-grid side length at the given (0-indexed) stage (patch_size=4)."""
    grid = img_size // 4
    for s in range(1, n_stages):
        if s <= stage:
            grid //= 2
    return grid


def _run_swin_hook(
    model: MILModel,
    frame_tensor: torch.Tensor,
    stage: int,
    block: int,
) -> tuple[torch.Tensor, tuple[int, int], int]:
    """
    Fire a forward pass through the Swin backbone with a hook on the
    target WindowAttention module.

    Returns
    ───────
    raw       : (nW*B, heads, N, N) attention tensor
    window_sz : (Wh, Ww)
    grid      : patch-grid side length at this stage
    """
    swin         = model.backbone
    n_stages     = len(swin.layers)
    actual_stage = stage if stage >= 0 else n_stages + stage

    target_block = swin.layers[stage].blocks[block]
    target_attn  = target_block.attn
    Wh, Ww       = target_attn.window_size

    captured   = {}
    orig_fused = target_attn.fused_attn
    target_attn.fused_attn = False  # expose explicit attn matrix

    def hook_fn(module, inp, out):
        captured["raw"] = _swin_attn_hook(module, inp, target_block)

    h = target_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        swin(frame_tensor)
    h.remove()
    target_attn.fused_attn = orig_fused

    grid = _grid_size_at_stage(model.cfg.img_size, n_stages, actual_stage)
    return captured["raw"], (Wh, Ww), grid


def _reassemble_windows(
    patch_imp: torch.Tensor,   # (nW*B, N)   column-mean attention
    Wh: int, Ww: int,
    grid: int,
) -> np.ndarray:
    """Tile window patches back into a (grid, grid) spatial map."""
    nH = max(1, grid // Wh)
    nW = max(1, grid // Ww)
    imp = patch_imp[: nH * nW]                          # (nH*nW, Wh*Ww)
    imp = imp.reshape(nH, nW, Wh, Ww)
    imp = imp.permute(0, 2, 1, 3).reshape(nH * Wh, nW * Ww)
    return imp[:grid, :grid].numpy()


def extract_swin_attention(
    model: MILModel,
    frame_tensor: torch.Tensor,
    stage: int = -1,
    block: int = -1,
) -> np.ndarray:
    """
    Head-averaged spatial attention map (grid_h, grid_w) from one Swin block.

    We hook model.backbone.layers[stage].blocks[block].attn, temporarily
    disable fused_attn so the explicit attention matrix is computed, then:
      1. Average over attention heads.
      2. Take the column-wise mean → how much each patch is attended to.
      3. Reassemble window tiles into the full patch grid.

    For the last stage at 384 px input (12×12 grid, 12×12 window) there is
    exactly one window, giving a 12×12 global attention map.
    """
    raw, (Wh, Ww), grid = _run_swin_hook(model, frame_tensor, stage, block)
    patch_imp = raw.mean(dim=1).mean(dim=1)             # (nW*B, N)
    return _reassemble_windows(patch_imp, Wh, Ww, grid)


def extract_swin_attention_multistage(
    model: MILModel, frame_tensor: torch.Tensor
) -> dict[int, np.ndarray]:
    """Return {stage_idx: attn_map} for the last block of every Swin stage."""
    result = {}
    for s in range(len(model.backbone.layers)):
        try:
            result[s] = extract_swin_attention(model, frame_tensor, stage=s, block=-1)
        except Exception as exc:
            print(f"  [WARN] Stage {s}: {exc}")
    return result


def extract_swin_perhead_attention(
    model: MILModel,
    frame_tensor: torch.Tensor,
    stage: int = -1,
    block: int = -1,
) -> np.ndarray:
    """
    Per-head spatial attention maps — shape (num_heads, grid_h, grid_w).
    Each slice is the column-mean attention received by every patch for that
    head, reassembled into the full patch grid.
    """
    raw, (Wh, Ww), grid = _run_swin_hook(model, frame_tensor, stage, block)
    num_heads = raw.shape[1]

    # per-head column mean → (nW*B, heads, N)
    per_head = raw.mean(dim=2)

    nH    = max(1, grid // Wh)
    nW    = max(1, grid // Ww)
    n_win = nH * nW

    imp = per_head[:n_win]                              # (nH*nW, heads, Wh*Ww)
    imp = imp.permute(1, 0, 2)                          # (heads, nH*nW, Wh*Ww)
    imp = imp.reshape(num_heads, nH, nW, Wh, Ww)
    imp = imp.permute(0, 1, 3, 2, 4).reshape(num_heads, nH * Wh, nW * Ww)

    return imp[:, :grid, :grid].numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap overlay
# ─────────────────────────────────────────────────────────────────────────────

def overlay_attention_heatmap(
    frame: np.ndarray,
    attn_map: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bilinearly upscale attn_map to frame size and alpha-blend.

    Args:
        frame   : (H, W) uint8 grayscale or (H, W, 3)
        attn_map: (grid_h, grid_w) float attention
        alpha   : heatmap blend weight
        cmap    : matplotlib colormap name
    Returns:
        overlay      : (H, W, 3) float32 blended image
        attn_upscaled: (H, W) float32 normalised attention
    """
    H, W = frame.shape[:2]
    lo, hi = attn_map.min(), attn_map.max()
    attn_norm = (attn_map - lo) / (hi - lo + 1e-8)

    attn_t  = torch.from_numpy(attn_norm).float().unsqueeze(0).unsqueeze(0)
    attn_up = F.interpolate(attn_t, size=(H, W), mode="bilinear",
                            align_corners=False).squeeze().numpy()

    heatmap = plt.cm.get_cmap(cmap)(attn_up)[:, :, :3]

    if frame.ndim == 2:
        frame_rgb = np.stack([frame] * 3, axis=-1).astype(np.float32) / 255.0
    else:
        frame_rgb = frame.astype(np.float32) / 255.0

    return np.clip((1 - alpha) * frame_rgb + alpha * heatmap, 0, 1), attn_up


# ─────────────────────────────────────────────────────────────────────────────
# Per-video analysis
# ─────────────────────────────────────────────────────────────────────────────

def to_gray(frame: np.ndarray) -> np.ndarray:
    return np.mean(frame, axis=-1).astype(np.uint8) if frame.ndim == 3 else frame


def build_bag(
    frames: np.ndarray, cfg: InferenceCFG, transform: T.Compose
) -> torch.Tensor:
    """Stack preprocessed frames into a (T, 1, H, W) bag tensor."""
    return torch.stack([
        transform(
            Image.fromarray(fr, mode="L") if fr.ndim == 2
            else Image.fromarray(fr).convert("L")
        )
        for fr in frames
    ]).to(cfg.device)


def analyse_video(
    model:      MILModel,
    cfg:        InferenceCFG,
    row:        pd.Series,
    transform:  T.Compose,
    out_dir:    Path,
    top_k:      int,
    vid_prefix: str,
) -> tuple[dict, np.ndarray, dict]:
    """
    Full single-video attention analysis.  Saves 6 figures and returns:
      (summary_dict, mil_alpha, swin_maps)
    so the caller can write the report without a second forward pass.
    """
    vid_id  = str(row[cfg.video_id_col])
    gt_pdff = float(row[cfg.target_col])

    print(f"\n{'─'*60}")
    print(f"  Video: {vid_id}  |  GT PDFF: {gt_pdff:.2f}%")

    # ── Load & sample frames ─────────────────────────────────────────────────
    all_frames  = _load_frames_for_row(row, cfg)
    sample_idxs = np.linspace(0, all_frames.shape[0] - 1,
                               num=cfg.n_frames, dtype=int)
    frames = all_frames[sample_idxs]
    bag    = build_bag(frames, cfg, transform)

    # ── MIL attention ────────────────────────────────────────────────────────
    pred_pdff, mil_alpha, _ = extract_mil_attention(model, bag)
    print(f"  Pred PDFF: {pred_pdff:.2f}%  |  MAE: {abs(pred_pdff - gt_pdff):.2f}%")

    N         = len(mil_alpha)
    uniform   = 1.0 / N
    top_k_idx = np.argsort(mil_alpha)[-top_k:][::-1]
    bot_k_idx = np.argsort(mil_alpha)[:top_k]

    entropy     = -np.sum(mil_alpha * np.log(mil_alpha + 1e-12))
    max_entropy = np.log(N)
    sorted_a    = np.sort(mil_alpha)[::-1]
    cumsum      = np.cumsum(sorted_a)
    n50 = int(np.searchsorted(cumsum, 0.5)) + 1
    n90 = int(np.searchsorted(cumsum, 0.9)) + 1

    # ── Fig 1: Attention timeline ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    x  = np.arange(N)
    ax.plot(x, mil_alpha, color="#2196F3", linewidth=0.8, alpha=0.8)
    ax.fill_between(x, mil_alpha, alpha=0.15, color="#2196F3")
    ax.scatter(top_k_idx, mil_alpha[top_k_idx], color="#F44336", s=40, zorder=5,
               edgecolors="white", linewidth=0.8, label=f"Top-{top_k}")
    ax.axhline(uniform, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Uniform")
    ax.set_ylabel("Attention Weight")
    ax.set_title(
        f"MIL Attention — {vid_id}  |  GT: {gt_pdff:.1f}%  |  "
        f"Pred: {pred_pdff:.1f}%  |  {type(model.aggregator).__name__}",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, N - 1)
    axes[1].imshow(mil_alpha[np.newaxis, :], aspect="auto", cmap="hot",
                   extent=[0, N, 0, 1])
    axes[1].set_xlabel("Frame Index")
    axes[1].set_yticks([])
    axes[1].set_ylabel("Attn")
    plt.tight_layout()
    savefig(fig, out_dir, f"{vid_prefix}_1_mil_attention_timeline")

    # ── Fig 2: Distribution + cumulative ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mil_alpha, bins=50, color="#2196F3", edgecolor="white",
                 linewidth=0.5, alpha=0.8)
    axes[0].axvline(uniform, color="red", linestyle="--", linewidth=1,
                    label="Uniform")
    axes[0].set_xlabel("Attention Weight")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Weight Distribution")
    axes[0].legend(fontsize=8)

    axes[1].plot(np.arange(1, N + 1), cumsum, color="#4CAF50", linewidth=1.5)
    for lvl, col in [(0.5, "#FF9800"), (0.9, "#F44336")]:
        axes[1].axhline(lvl, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].axvline(n50, color="#FF9800", linestyle=":", linewidth=1,
                    label=f"50% → {n50} frames")
    axes[1].axvline(n90, color="#F44336", linestyle=":", linewidth=1,
                    label=f"90% → {n90} frames")
    axes[1].set_xlabel("Top-N Frames")
    axes[1].set_ylabel("Cumulative Attention")
    axes[1].set_title("Attention Concentration")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    savefig(fig, out_dir, f"{vid_prefix}_2_mil_distribution")

    # ── Fig 3: Top-K vs Bottom-K frames ─────────────────────────────────────
    fig, axes = plt.subplots(2, top_k, figsize=(2.5 * top_k, 6))
    for col, idx in enumerate(top_k_idx):
        fr = to_gray(frames[idx])
        axes[0, col].imshow(fr, cmap="gray")
        axes[0, col].set_title(f"#{idx}\n{mil_alpha[idx]:.4f}", fontsize=8,
                               fontweight="bold", color="#D32F2F")
        axes[0, col].axis("off")
    for col, idx in enumerate(bot_k_idx):
        fr = to_gray(frames[idx])
        axes[1, col].imshow(fr, cmap="gray")
        axes[1, col].set_title(f"#{idx}\n{mil_alpha[idx]:.6f}", fontsize=8,
                               color="#1565C0")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("High\nAttn", fontsize=10, fontweight="bold",
                           rotation=0, labelpad=50, va="center")
    axes[1, 0].set_ylabel("Low\nAttn",  fontsize=10, fontweight="bold",
                           rotation=0, labelpad=50, va="center")
    fig.suptitle(f"Top vs Bottom {top_k} Frames — {vid_id}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, out_dir, f"{vid_prefix}_3_mil_top_bottom_frames")

    # ── Swin attention for top-K frames ─────────────────────────────────────
    print(f"  Extracting Swin attention for top-{top_k} frames …")
    swin_maps = {}
    for frame_idx in top_k_idx:
        swin_maps[frame_idx] = extract_swin_attention(
            model, bag[frame_idx: frame_idx + 1], stage=-1, block=-1
        )

    # ── Fig 4: Heatmap overlays ──────────────────────────────────────────────
    n_show = len(swin_maps)
    fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
    if n_show == 1:
        axes = axes[:, np.newaxis]
    for col, frame_idx in enumerate(top_k_idx[:n_show]):
        fr       = to_gray(frames[frame_idx])
        attn_map = swin_maps[frame_idx]
        overlay, _ = overlay_attention_heatmap(fr, attn_map)

        axes[0, col].imshow(fr, cmap="gray")
        axes[0, col].set_title(f"Frame #{frame_idx}\nα={mil_alpha[frame_idx]:.4f}",
                               fontsize=8, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title("Swin Attn\n(Last Stage)", fontsize=8)
        axes[1, col].axis("off")

        axes[2, col].imshow(attn_map, cmap="hot", interpolation="nearest")
        axes[2, col].set_title(
            f"Raw ({attn_map.shape[0]}×{attn_map.shape[1]})", fontsize=8
        )
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight="bold",
                           rotation=0, labelpad=50, va="center")
    axes[1, 0].set_ylabel("Overlay",  fontsize=10, fontweight="bold",
                           rotation=0, labelpad=50, va="center")
    axes[2, 0].set_ylabel("Attn Map", fontsize=10, fontweight="bold",
                           rotation=0, labelpad=50, va="center")
    fig.suptitle(
        f"Swin Backbone Heatmaps — Top {n_show} MIL Frames\n"
        f"{vid_id}  |  GT: {gt_pdff:.1f}%  |  Pred: {pred_pdff:.1f}%",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, out_dir, f"{vid_prefix}_4_swin_heatmap_overlay")

    # ── Fig 5: Multi-stage attention (best frame) ────────────────────────────
    best_idx    = top_k_idx[0]
    best_tensor = bag[best_idx: best_idx + 1]
    best_frame  = to_gray(frames[best_idx])

    ms_attn    = extract_swin_attention_multistage(model, best_tensor)
    stage_info = {
        0: "96×96 3h\nTexture",
        1: "48×48 6h\nLocal",
        2: "24×24 12h\nMid",
        3: "12×12 24h\nGlobal",
    }
    n_cols = len(ms_attn) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

    axes[0, 0].imshow(best_frame, cmap="gray")
    axes[0, 0].set_title(f"Original\nFrame #{best_idx}", fontsize=9, fontweight="bold")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for off, (s_idx, attn_map) in enumerate(sorted(ms_attn.items())):
        c = off + 1
        overlay, _ = overlay_attention_heatmap(best_frame, attn_map)
        axes[0, c].imshow(overlay)
        axes[0, c].set_title(f"Stage {s_idx}\n{stage_info.get(s_idx, '')}",
                             fontsize=8)
        axes[0, c].axis("off")
        axes[1, c].imshow(attn_map, cmap="hot", interpolation="nearest")
        axes[1, c].set_title(f"{attn_map.shape[0]}×{attn_map.shape[1]}", fontsize=8)
        axes[1, c].axis("off")

    fig.suptitle(
        f"Multi-Stage Swin Attention — Frame #{best_idx}\n"
        f"{vid_id}  |  GT: {gt_pdff:.1f}%  |  Pred: {pred_pdff:.1f}%",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, out_dir, f"{vid_prefix}_5_swin_multistage")

    # ── Fig 6: Per-head attention (last stage) ───────────────────────────────
    perhead      = extract_swin_perhead_attention(model, best_tensor,
                                                  stage=-1, block=-1)
    n_heads_show = min(12, perhead.shape[0])
    head_var     = perhead.reshape(perhead.shape[0], -1).var(axis=1)
    top_heads    = np.argsort(head_var)[-n_heads_show:][::-1]

    ncols = min(6, n_heads_show)
    nrows = (n_heads_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i, h_idx in enumerate(top_heads):
        r, c = divmod(i, ncols)
        overlay, _ = overlay_attention_heatmap(best_frame, perhead[h_idx], alpha=0.55)
        axes[r, c].imshow(overlay)
        axes[r, c].set_title(f"Head {h_idx}\nvar={head_var[h_idx]:.4f}", fontsize=7)
        axes[r, c].axis("off")
    for i in range(n_heads_show, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Per-Head Swin Attention (Last Stage) — Top {n_heads_show} by Variance\n"
        f"Frame #{best_idx}  |  {vid_id}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, out_dir, f"{vid_prefix}_6_swin_perhead")

    summary = {
        "video":         vid_id,
        "gt_pdff":       gt_pdff,
        "pred_pdff":     pred_pdff,
        "mae":           abs(pred_pdff - gt_pdff),
        "attn_max":      mil_alpha.max(),
        "attn_std":      mil_alpha.std(),
        "n50":           n50,
        "n90":           n90,
        "entropy_ratio": entropy / max_entropy,
    }
    return summary, mil_alpha, swin_maps


# ─────────────────────────────────────────────────────────────────────────────
# Batch summary figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_batch_summary(results_df: pd.DataFrame, n_frames: int, out_dir: Path) -> None:
    if len(results_df) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    sc = ax.scatter(results_df["gt_pdff"], results_df["entropy_ratio"],
                    c=results_df["mae"], cmap="RdYlGn_r", s=60,
                    edgecolors="black", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="MAE")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Ground Truth PDFF (%)")
    ax.set_ylabel("Attention Entropy Ratio")
    ax.set_title("Attention Concentration vs PDFF")

    ax = axes[1]
    ax.bar(range(len(results_df)), results_df["n50"], color="#2196F3", alpha=0.7)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df["video"], rotation=45, ha="right", fontsize=7)
    ax.axhline(n_frames / 2, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel(f"Frames for 50% Attn (of {n_frames})")
    ax.set_title("Attention Sparsity per Video")

    ax = axes[2]
    ax.scatter(results_df["gt_pdff"], results_df["pred_pdff"],
               c=results_df["entropy_ratio"], cmap="coolwarm", s=60,
               edgecolors="black", linewidth=0.5)
    lim = max(results_df["gt_pdff"].max(), results_df["pred_pdff"].max()) + 2
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Ground Truth PDFF (%)")
    ax.set_ylabel("Predicted PDFF (%)")
    ax.set_title("Predictions (colour = entropy ratio)")
    ax.set_aspect("equal")
    ax.set_xlim(-1, lim)
    ax.set_ylim(-1, lim)

    plt.tight_layout()
    savefig(fig, out_dir, "batch_summary")


# ─────────────────────────────────────────────────────────────────────────────
# Interpretability report
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_report(
    cfg:       InferenceCFG,
    model:     MILModel,
    ckpt_path: Path,
    vid_id:    str,
    gt_pdff:   float,
    pred_pdff: float,
    mil_alpha: np.ndarray,
    top_k_idx: np.ndarray,
    swin_maps: dict,
    out_dir:   Path,
) -> None:
    n     = len(mil_alpha)
    e     = -np.sum(mil_alpha * np.log(mil_alpha + 1e-12))
    e_max = np.log(n)
    ratio = e / e_max

    lines = [
        "=" * 70,
        "ATTENTION ANALYSIS REPORT",
        "=" * 70,
        "",
        "Model",
        f"  Backbone    : {cfg.backbone}",
        f"  Aggregator  : {cfg.aggregator}  ({type(model.aggregator).__name__})",
        f"  Image size  : {cfg.img_size}×{cfg.img_size}",
        f"  Frames/bag  : {cfg.n_frames}",
        f"  Checkpoint  : {ckpt_path}",
        "",
        "MIL-Level Attention",
        f"  Video       : {vid_id}",
        f"  GT PDFF     : {gt_pdff:.2f}%",
        f"  Pred PDFF   : {pred_pdff:.2f}%",
        f"  Abs error   : {abs(pred_pdff - gt_pdff):.2f}%",
        f"  Alpha range : [{mil_alpha.min():.6f}, {mil_alpha.max():.6f}]",
        f"  Max/Uniform : {mil_alpha.max() * n:.1f}×",
        f"  Entropy     : {e:.3f} / {e_max:.3f}  (ratio {ratio:.3f})",
    ]
    if ratio > 0.95:
        lines.append("  → UNIFORM attention — model distributes focus broadly")
    elif ratio > 0.80:
        lines.append("  → MODERATE concentration — some temporal selectivity")
    else:
        lines.append("  → HIGH concentration — strong temporal selectivity")

    lines += ["", f"  Top-{len(top_k_idx)} frames:"]
    for rank, idx in enumerate(top_k_idx):
        lines.append(f"    #{rank+1}: frame {idx:4d}  "
                     f"(t={idx/n*100:5.1f}%)  α={mil_alpha[idx]:.6f}")

    lines += [
        "",
        "Backbone (Swin) Spatial Attention",
        "  Hook : model.backbone.layers[-1].blocks[-1].attn  (WindowAttention)",
        f"  Grid : {_grid_size_at_stage(cfg.img_size, 4, 3)}×"
        f"{_grid_size_at_stage(cfg.img_size, 4, 3)} patches, "
        f"{model.backbone.layers[-1].blocks[-1].attn.num_heads} heads",
    ]

    if swin_maps:
        top_attn = swin_maps[top_k_idx[0]]
        norm     = (top_attn - top_attn.min()) / (top_attn.max() - top_attn.min() + 1e-8)
        mask     = norm > 0.7
        if mask.any():
            rows, cols = np.where(mask)
            cy, cx = rows.mean(), cols.mean()
            h, w   = top_attn.shape
            ry = "top"    if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
            rx = "left"   if cx < w / 3 else ("right"  if cx > 2 * w / 3 else "center")
            lines += [
                f"  High-attn region (>70% of max):",
                f"    Location : {ry}-{rx}",
                f"    Coverage : {100*mask.sum()/mask.size:.1f}% of patch grid",
                f"    Centroid : ({cx:.1f}, {cy:.1f}) in {w}×{h} grid",
            ]
        else:
            lines.append("  Attention is spatially diffuse (no patch >70% of max)")

    lines += [
        "",
        "Clinical Notes",
        "  • High-MIL frames = most diagnostic liver-parenchyma views in sweep.",
        "  • Swin heatmap = spatial regions driving fat-fraction estimation.",
        "  • Low entropy → model learned specific sweep positions over the liver.",
        "  • Diffuse spatial attn → global texture pattern (typical for US fat).",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print(report)
    (out_dir / "report.txt").write_text(report)
    print(f"  Saved: report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attention visualisation for the EUS liver PDFF MIL model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpt", required=True, type=Path,
        help="Path to best.pt, e.g. runs_with_test/mh_gated4_s384_f175/fold00/best.pt",
    )
    parser.add_argument(
        "--sample_idx", type=int, default=0,
        help="Index into the validation split for single-video mode (default: 0)",
    )
    parser.add_argument(
        "--top_k", type=int, default=8,
        help="Number of high-attention frames to visualise (default: 8)",
    )
    parser.add_argument(
        "--all_videos", action="store_true",
        help="Run analysis on every validation video and save a batch summary",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    ckpt_path = args.ckpt.resolve()
    fold_dir  = ckpt_path.parent

    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # ── Load config.json ──────────────────────────────────────────────────────
    config_path = find_config_json(ckpt_path)
    print(f"\nCheckpoint : {ckpt_path}")
    print(f"Config     : {config_path}")

    cfg         = InferenceCFG(config_path, args.device)
    cfg.set_col = resolve_set_col(cfg, fold_dir)

    print(f"Aggregator : {cfg.aggregator}")
    print(f"Backbone   : {cfg.backbone}")
    print(f"img_size   : {cfg.img_size}  |  n_frames: {cfg.n_frames}")
    print(f"CSV        : {cfg.csv_path}")
    print(f"Split col  : {cfg.set_col}")

    out_dir = fold_dir / "attention_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output     : {out_dir}")

    set_seed(cfg.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    model = MILModel(cfg).to(args.device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=args.device, weights_only=True)
    )
    model.eval()
    print(f"Loaded     : {type(model.aggregator).__name__}  "
          f"({model.backbone.num_features}d features)")

    # ── Load split ────────────────────────────────────────────────────────────
    df     = pd.read_csv(cfg.csv_path)
    val_df = df[df[cfg.set_col] == "val"].reset_index(drop=True)
    if val_df.empty:
        # fall back to train split if val is empty (e.g. runs_with_test setup)
        val_df = df[df[cfg.set_col] == "train"].reset_index(drop=True)
        print(f"[WARN] No 'val' rows in {cfg.set_col}; falling back to 'train' split.")
    print(f"Videos     : {len(val_df)}")

    transform = T.Compose([T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor()])

    if args.all_videos:
        # ── Batch mode ────────────────────────────────────────────────────────
        all_results = []
        for i in range(len(val_df)):
            row    = val_df.iloc[i]
            vid_id = str(row[cfg.video_id_col])
            vid_dir = out_dir / vid_id
            vid_dir.mkdir(parents=True, exist_ok=True)
            try:
                summary, mil_alpha, swin_maps = analyse_video(
                    model, cfg, row, transform, vid_dir, args.top_k, vid_id
                )
                top_k_idx = np.argsort(mil_alpha)[-args.top_k:][::-1]
                print_and_save_report(
                    cfg, model, ckpt_path,
                    vid_id, float(row[cfg.target_col]), summary["pred_pdff"],
                    mil_alpha, top_k_idx, swin_maps, vid_dir,
                )
                all_results.append(summary)
            except Exception as exc:
                print(f"  [ERROR] {vid_id}: {exc}")

        if all_results:
            results_df = pd.DataFrame(all_results)
            csv_path   = out_dir / "batch_attention_metrics.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"\nBatch CSV  : {csv_path}")

            plot_batch_summary(results_df, cfg.n_frames, out_dir)

            print(f"\nMean MAE          : {results_df['mae'].mean():.2f} "
                  f"± {results_df['mae'].std():.2f}")
            print(f"Mean entropy ratio: {results_df['entropy_ratio'].mean():.3f}")
            print(f"Mean n50          : {results_df['n50'].mean():.0f}/{cfg.n_frames}")

    else:
        # ── Single-video mode ─────────────────────────────────────────────────
        assert args.sample_idx < len(val_df), (
            f"--sample_idx {args.sample_idx} out of range "
            f"(split '{cfg.set_col}' has {len(val_df)} videos)"
        )
        row    = val_df.iloc[args.sample_idx]
        vid_id = str(row[cfg.video_id_col])
        vid_dir = out_dir / vid_id
        vid_dir.mkdir(parents=True, exist_ok=True)

        summary, mil_alpha, swin_maps = analyse_video(
            model, cfg, row, transform, vid_dir, args.top_k, vid_id
        )
        top_k_idx = np.argsort(mil_alpha)[-args.top_k:][::-1]
        print_and_save_report(
            cfg, model, ckpt_path,
            vid_id, float(row[cfg.target_col]), summary["pred_pdff"],
            mil_alpha, top_k_idx, swin_maps, vid_dir,
        )

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
