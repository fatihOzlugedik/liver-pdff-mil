# model.py — MIL model with multiple pooling strategies for video regression
import re, torch, timm
import torch.nn as nn
from config import CFG

# ---------------------------- poolers ---------------------------------
class MeanPool(nn.Module):
    def forward(self, x):  # x: (inst, D)
        return x.mean(0)

class MaxPool(nn.Module):
    def forward(self, x):
        return x.max(0).values

class AttentionMIL(nn.Module):
    """Single-head additive attention (Ilse et al.)."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.att = nn.Sequential(
            
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
    def forward(self, x):                      # x: (inst, D)
        alpha = torch.softmax(self.att(x), 0)  # (inst,1)
        return (alpha * x).sum(0)              # (D,)

class GatedAttentionMIL(nn.Module):
    """Gated attention (Ilse et al. Eq 10)."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
    def forward(self, x):
        a = torch.sigmoid(self.U(x)) * torch.tanh(self.V(x))
        alpha = torch.softmax(self.w(a), 0)
        return (alpha * x).sum(0)

class MultiHeadGatedAttn(nn.Module):
    """k independent gated heads + concat + linear."""
    def __init__(self, dim, heads=4, hidden=128):
        super().__init__()
        self.heads = nn.ModuleList([GatedAttentionMIL(dim, hidden)
                                    for _ in range(heads)])
        self.proj  = nn.Linear(dim * heads, dim)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class TemporalConvGatedMIL(nn.Module):
    """
    1-D depthwise-separable conv along time ➜ per-frame features
    ➜ Multi-Head Gated Attention pooling.
    """
    def __init__(self, dim, out_ch=16, k=5, heads=4, hidden=128):
        super().__init__()
        pad = k // 2
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, k, padding=pad, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, out_ch, 1),
            nn.GELU(),
            nn.Conv1d(out_ch, dim, 1)
        )
        self.pool = MultiHeadGatedAttn(dim, heads=heads, hidden=hidden)
    def forward(self, x):                      # x: (inst, D)
        y = self.conv(x.T.unsqueeze(0))        # (1, D, inst)
        feats = y.squeeze(0).T                 # (inst, D)
        return self.pool(feats)                # (D,)

class ABMILPool(nn.Module):
    """
    ABMIL (Attention-Based Deep MIL) pooler — adapted from MIL-Lab.

    Architecture (Ilse et al., ICML 2018):
      1. patch_embed MLP: maps each instance D → embed_dim (nonlinear projection
         with ReLU + dropout, giving the attention mechanism a learned subspace).
      2. Gated attention: Tanh(Wh) ⊙ Sigmoid(Uh) → linear → scalar score per instance.
         The sigmoid gate controls information flow, suppressing noisy instances.
      3. Softmax over instances → weighted sum → bag-level feature (embed_dim).
      4. proj_back: linear embed_dim → D to match the regressor input dimension.

    Differences vs existing GatedAttentionMIL in this file:
      - Adds a patch_embed MLP before attention (learnable subspace projection).
      - Adds dropout inside both the MLP and the attention branches.
      - Uses Kaiming initialization (important for deeper attention networks).

    Reference:
      Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance Learning",
      ICML 2018.
    """
    def __init__(self, dim, embed_dim=512, attn_dim=384, dropout=0.25):
        super().__init__()

        # Patch embedding: nonlinear projection of each instance feature
        self.patch_embed = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention: Tanh path ⊙ Sigmoid path → linear → score
        self.attn_a = nn.Sequential(
            nn.Linear(embed_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attn_b = nn.Sequential(
            nn.Linear(embed_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.attn_c = nn.Linear(attn_dim, 1)

        # Project back to original dim so the regressor interface is unchanged
        self.proj_back = nn.Linear(embed_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):                          # x: (inst, D)
        h = self.patch_embed(x)                    # (inst, embed_dim)
        A = self.attn_a(h) * self.attn_b(h)        # (inst, attn_dim)
        A = self.attn_c(A)                         # (inst, 1)
        A = torch.softmax(A, dim=0)                # normalize over instances
        pooled = (A * h).sum(0)                    # (embed_dim,)
        return self.proj_back(pooled)              # (D,)


# ----------------------------- factory ---------------------------------
def build_pooler(name: str, dim: int):
    """
    Build a MIL pooler by name.

    Recognized names:
      - "mean": Simple mean pooling
      - "max": Max pooling
      - "attention": Single-head additive attention (Ilse et al.)
      - "gated": Single-head gated attention
      - "abmil", "abmil{E}": ABMIL with optional embed_dim (e.g., abmil512)
      - "mh_gated{K}": Multi-head gated attention with K heads
      - "temporal_conv{out}h{heads}": Temporal conv + multi-head gated

    Examples:
      build_pooler("mean", 768)
      build_pooler("abmil", 768)
      build_pooler("abmil512", 768)
      build_pooler("mh_gated4", 768)
      build_pooler("temporal_conv32h4", 768)
    """
    name = name.lower()

    if name == "mean":
        return MeanPool()
    if name == "max":
        return MaxPool()
    if name == "attention":
        return AttentionMIL(dim)
    if name == "gated":
        return GatedAttentionMIL(dim)

    # ------------ ABMIL (MIL-Lab) ------------------
    # "abmil" (default dims), "abmil512", "abmil256" (custom embed_dim)
    if name.startswith("abmil"):
        nums = list(map(int, re.findall(r"\d+", name)))
        embed = nums[0] if nums else 512
        return ABMILPool(dim, embed_dim=embed)

    # ------------ multi-head gated -----------------
    if name.startswith("mh_gated"):
        digits = re.findall(r"\d+", name)
        k = int(digits[0]) if digits else 4
        return MultiHeadGatedAttn(dim, heads=k)

    # ------------ temporal conv → gated ------------
    if name.startswith("temporal_conv"):
        nums = list(map(int, re.findall(r"\d+", name)))
        out_ch = nums[0] if nums else 16
        heads  = nums[1] if len(nums) > 1 else 4
        return TemporalConvGatedMIL(dim, out_ch=out_ch, heads=heads)

    raise ValueError(f"Unknown aggregator '{name}'. "
                     f"Available: mean, max, attention, gated, abmil, abmil{{E}}, mh_gated{{K}}, temporal_conv{{out}}h{{heads}}")

# --------------------------- main model --------------------------------
class MILModel(nn.Module):
    """
    CNN/ViT backbone ➜ MIL pooler ➜ regressor + optional classifier.

    Multi-task learning: combines regression (PDFF %) with classification
    (4-class PDFF staging) using a shared feature backbone.
    """
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.use_classifier = getattr(cfg, "use_classifier", False)
        self.num_classes = getattr(cfg, "num_classes", 4)

        # For Swin at non-native resolution, window_size must divide the patch grid.
        # e.g. swin_tiny 384: patch grid=24x24, need window_size=12 (not default 7).
        # Set cfg.window_size > 0 to override, otherwise use model default.
        extra_kwargs = {}
        if cfg.img_size != 224 and "swin" in cfg.backbone.lower():
            # Swin total stride=32, so final grid = img_size//32; window = that value
            # e.g. 384//32=12, 224//32=7
            auto_window = cfg.img_size // 32
            resolved_window = getattr(cfg, "window_size", 0) or auto_window
        elif getattr(cfg, "window_size", 0) > 0:
            resolved_window = cfg.window_size
        else:
            resolved_window = 0

        if resolved_window > 0:
            extra_kwargs["window_size"] = resolved_window
            cfg.window_size = resolved_window  # write back so config.json is accurate

        self.backbone = timm.create_model(
            cfg.backbone, pretrained=True, in_chans=1, num_classes=0,
            img_size=cfg.img_size, **extra_kwargs
        )
        dim = self.backbone.num_features
        self.aggregator = build_pooler(cfg.aggregator, dim)

        # Regression head: predicts PDFF percentage
        self.regressor = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1)
        )

        # Classification head: predicts PDFF stage (4 classes)
        if self.use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, self.num_classes)
            )

    def forward(self, bag: torch.Tensor):
        """
        Args:
            bag: (T, C, H, W) tensor of video frames

        Returns:
            If use_classifier=False: regression output (1,)
            If use_classifier=True: tuple of (regression (1,), classification logits (num_classes,))
        """
        feats = self.backbone(bag)          # (T, D)
        pooled = self.aggregator(feats)     # (D,)

        reg_out = self.regressor(pooled)    # (1,)

        if self.use_classifier:
            cls_out = self.classifier(pooled)  # (num_classes,)
            return reg_out, cls_out

        return reg_out
