# model.py — MIL-Lab model wrappers for PDFF regression
#
# Architecture per model:
#   Swin-Tiny backbone  ->  features [T, D]
#   MIL-Lab aggregator  ->  pooled   [embed_dim]
#   Regression head     ->  scalar   PDFF prediction
#
# Supported aggregators: abmil, transmil, clamsb, dsmil

import torch
import torch.nn as nn
import timm
from config import CFG

# Local cleaned copies of MIL-Lab models (no transformers dependency)
from millab_models import ABMIL, TransMIL, CLAMSB, DSMIL


# ── Aggregator factory ───────────────────────────────────────────────────

def build_aggregator(name: str, backbone_dim: int, embed_dim: int = 512):
    """
    Build a MIL-Lab aggregator module.

    All aggregators expose .forward_features(h) -> (pooled, log_dict)
    where h is [B, M, D] and pooled is [B, embed_dim].

    For DSMIL, pooled shape is [B, 1, embed_dim] (num_classes=1).
    """
    name = name.lower()

    if name == "abmil":
        return ABMIL(
            in_dim=backbone_dim,
            embed_dim=embed_dim,
            num_fc_layers=1,
            dropout=0.25,
            attn_dim=384,
            gate=True,
            num_classes=1,  # head unused; 1 keeps attention single-head
        )

    if name == "transmil":
        return TransMIL(
            in_dim=backbone_dim,
            embed_dim=embed_dim,
            num_fc_layers=1,
            dropout=0.25,
            num_attention_layers=2,
            num_heads=8,
            num_classes=1,  # head unused
        )

    if name == "clamsb":
        return CLAMSB(
            in_dim=backbone_dim,
            embed_dim=embed_dim,
            n_fc_layers=1,
            dropout=0.25,
            gate=True,
            attention_dim=384,
            num_classes=2,          # needed for instance classifiers
            k_sample=8,
            subtyping=False,
            instance_loss_fn="ce",  # avoid SmoothTop1SVM dependency
            bag_weight=0.7,
        )

    if name == "dsmil":
        return DSMIL(
            in_dim=backbone_dim,
            embed_dim=embed_dim,
            num_fc_layers=1,
            dropout=0.25,
            attn_dim=384,
            dropout_v=0.0,
            num_classes=1,  # single output stream for regression
        )

    raise ValueError(f"Unknown MIL-Lab aggregator '{name}'. "
                     f"Choose from: abmil, transmil, clamsb, dsmil")


# ── Main model ───────────────────────────────────────────────────────────

class MILLabModel(nn.Module):
    """
    Swin-Tiny backbone -> MIL-Lab aggregator -> regression head.

    The backbone extracts per-frame features.  The MIL-Lab model pools them
    into a single bag-level representation via forward_features().  A small
    regression MLP maps the pooled vector to a scalar PDFF prediction.
    """

    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.agg_name = cfg.aggregator.lower()

        # ── Backbone ─────────────────────────────────────────────────
        extra_kwargs = {}
        if cfg.img_size != 224 and "swin" in cfg.backbone.lower():
            auto_window = cfg.img_size // 32
            resolved_window = getattr(cfg, "window_size", 0) or auto_window
        elif getattr(cfg, "window_size", 0) > 0:
            resolved_window = cfg.window_size
        else:
            resolved_window = 0

        if resolved_window > 0:
            extra_kwargs["window_size"] = resolved_window
            cfg.window_size = resolved_window

        self.backbone = timm.create_model(
            cfg.backbone, pretrained=True, in_chans=1, num_classes=0,
            img_size=cfg.img_size, **extra_kwargs
        )
        backbone_dim = self.backbone.num_features  # 768 for swin_tiny

        # ── MIL-Lab aggregator ───────────────────────────────────────
        embed_dim = cfg.embed_dim
        self.aggregator = build_aggregator(cfg.aggregator, backbone_dim, embed_dim)

        # ── Regression head ──────────────────────────────────────────
        # DSMIL forward_features returns [B, 1, embed_dim]; others return [B, embed_dim]
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bag: (T, C, H, W) — T frames of a single video.
        Returns:
            scalar PDFF prediction (shape [1]).
        """
        # Extract per-frame features
        feats = self.backbone(bag)          # (T, D)
        feats = feats.unsqueeze(0)          # (1, T, D) — batch dim for MIL-Lab

        # Aggregate via MIL-Lab model
        pooled, _ = self.aggregator.forward_features(feats)  # (1, embed_dim) or (1, 1, embed_dim)

        # Flatten any extra dims (DSMIL returns [B, C, D] with C=1)
        if pooled.dim() == 3:
            pooled = pooled.squeeze(1)      # (1, embed_dim)
        pooled = pooled.squeeze(0)          # (embed_dim,)

        return self.regressor(pooled)       # (1,)
