# model.py — Drop-in MIL with TransMIL-Temporal for video (~700 frames), no chunking
import re, math, torch, timm
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
      2. Gated attention (default) or standard additive attention:
         - Gated: Tanh(Wh) ⊙ Sigmoid(Uh) → linear → scalar score per instance.
           The sigmoid gate controls information flow, suppressing noisy instances.
         - Non-gated: Tanh(Wh) → linear → scalar score per instance.
      3. Softmax over instances → weighted sum → bag-level feature (embed_dim).
      4. proj_back: linear embed_dim → D to match the regressor input dimension.

    Differences vs existing GatedAttentionMIL in this file:
      - Adds a patch_embed MLP before attention (learnable subspace projection).
      - Adds dropout inside both the MLP and the attention branches.
      - Uses Kaiming initialization (important for deeper attention networks).
      - Supports both gated and non-gated variants via the 'gate' flag.

    Reference:
      Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance Learning",
      ICML 2018.  MIL-Lab implementation: src/MIL-Lab/src/models/abmil.py
    """
    def __init__(self, dim, embed_dim=512, attn_dim=384, dropout=0.25, gate=True):
        super().__init__()
        self.gate = gate

        # Patch embedding: nonlinear projection of each instance feature
        self.patch_embed = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention mechanism
        if gate:
            # Gated: Tanh path ⊙ Sigmoid path → linear → score
            self.attn_a = nn.Sequential(
                nn.Linear(embed_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout)
            )
            self.attn_b = nn.Sequential(
                nn.Linear(embed_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout)
            )
            self.attn_c = nn.Linear(attn_dim, 1)
        else:
            # Non-gated: Tanh → linear → score
            self.attn = nn.Sequential(
                nn.Linear(embed_dim, attn_dim), nn.Tanh(),
                nn.Dropout(dropout), nn.Linear(attn_dim, 1),
            )

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
        if self.gate:
            A = self.attn_a(h) * self.attn_b(h)   # (inst, attn_dim)
            A = self.attn_c(A)                     # (inst, 1)
        else:
            A = self.attn(h)                       # (inst, 1)
        A = torch.softmax(A, dim=0)                # normalize over instances
        pooled = (A * h).sum(0)                    # (embed_dim,)
        return self.proj_back(pooled)              # (D,)


class TemporalABMILPool(nn.Module):
    """
    Temporal ABMIL — ABMIL with temporal positional encoding (TPEG).

    Injects multi-scale temporal context (via TPEG) into frame features
    BEFORE the ABMIL attention mechanism. This lets the gated attention
    know *where* in the video each frame comes from, so it can learn
    temporally-dependent importance patterns (e.g. "mid-sweep frames
    are more diagnostic").

    Architecture:
      1. TPEG: residual multi-scale 1-D depthwise conv (k=3,5,7) → fuse → D.
         Adds temporal position info to each frame's feature vector.
      2. patch_embed MLP: D → embed_dim (same as standard ABMIL).
      3. Gated attention → softmax → weighted sum (same as standard ABMIL).
      4. proj_back: embed_dim → D.

    This is the temporal counterpart of ABMILPool — identical except for
    the TPEG injection. Comparing abmil vs tabmil isolates the effect of
    temporal position information on attention-based frame weighting.
    """
    def __init__(self, dim, embed_dim=512, attn_dim=384, dropout=0.25, gate=True):
        super().__init__()
        self.gate = gate

        # Temporal positional encoding (residual, applied to raw features)
        self.tpeg = TPEG(dim)

        # Patch embedding: nonlinear projection of each instance feature
        self.patch_embed = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention mechanism (identical to ABMILPool)
        if gate:
            self.attn_a = nn.Sequential(
                nn.Linear(embed_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout)
            )
            self.attn_b = nn.Sequential(
                nn.Linear(embed_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout)
            )
            self.attn_c = nn.Linear(attn_dim, 1)
        else:
            self.attn = nn.Sequential(
                nn.Linear(embed_dim, attn_dim), nn.Tanh(),
                nn.Dropout(dropout), nn.Linear(attn_dim, 1),
            )

        self.proj_back = nn.Linear(embed_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):                          # x: (inst, D)
        x = x + self.tpeg(x)                      # residual temporal PE
        h = self.patch_embed(x)                    # (inst, embed_dim)
        if self.gate:
            A = self.attn_a(h) * self.attn_b(h)   # (inst, attn_dim)
            A = self.attn_c(A)                     # (inst, 1)
        else:
            A = self.attn(h)                       # (inst, 1)
        A = torch.softmax(A, dim=0)                # normalize over instances
        pooled = (A * h).sum(0)                    # (embed_dim,)
        return self.proj_back(pooled)              # (D,)


class GRUPool(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.proj= nn.Linear(hidden, dim)
    def forward(self, x):              # (inst,D)
        _, h = self.gru(x.unsqueeze(0))
        return self.proj(h.squeeze(0))

class TinyTransformerPool(nn.Module):
    """Single Transformer encoder block, then mean."""
    def __init__(self, dim, heads=4, ff=512):
        super().__init__()
        self.enc = nn.TransformerEncoderLayer(dim, heads, ff, batch_first=True)
    def forward(self, x):
        return self.enc(x.unsqueeze(0)).mean(1).squeeze(0)

# ----------------------- TransMIL (2-D grid) components ----------------
# Kept for completeness (not used for videos).
class PPEG(nn.Module):
    """Pyramid Positional Encoding Generator (2-D, depthwise 3/5/7)."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.proj  = nn.Linear(dim * 3, dim)
        self.act   = nn.GELU()
    def forward(self, tokens: torch.Tensor, N: int) -> torch.Tensor:
        D = tokens.shape[1]                               # tokens: (N*N, D)
        img = tokens.transpose(0, 1).reshape(1, D, N, N) # (1,D,N,N)
        y3 = self.conv3(img); y5 = self.conv5(img); y7 = self.conv7(img)
        y  = torch.cat([y3, y5, y7], dim=1)              # (1,3D,N,N)
        y  = y.permute(0, 2, 3, 1).reshape(1, N*N, 3*D)  # (1,N^2,3D)
        y  = self.proj(y)                                # (1,N^2,D)
        return self.act(y.squeeze(0))                    # (N^2,D)

class TransNILPool(nn.Module):
    """2-D TransMIL-style pooling with optional PPEG."""
    def __init__(self, dim, heads=4, depth=2, ff_mult=4, dropout=0.1, use_ppeg=True):
        super().__init__()
        self.use_ppeg = use_ppeg
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff_mult * dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm_out = nn.LayerNorm(dim)
        self.ppeg = PPEG(dim) if use_ppeg else None
    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (inst,D)
        n, d = x.shape
        device = x.device
        N = int(math.ceil(math.sqrt(max(n, 1))))
        pad_len = N * N - n
        if pad_len > 0:
            x = torch.cat([x, x.new_zeros(pad_len, d)], dim=0)  # (N^2,D)
        if self.ppeg is not None:
            x = x + self.ppeg(x, N)                             # residual PPEG
        x   = x.view(1, N * N, d)                               # (1,N^2,D)
        cls = self.cls_token.expand(1, -1, -1).to(device)       # (1,1,D)
        src = torch.cat([cls, x], dim=1)                        # (1,1+N^2,D)
        key_padding_mask = None
        if pad_len > 0:
            key_padding_mask = torch.zeros((1, 1 + N * N), dtype=torch.bool, device=device)
            key_padding_mask[0, 1 + n: 1 + N * N] = True
        out = self.encoder(src, src_key_padding_mask=key_padding_mask)  # (1,S,D)
        return self.norm_out(out[:, 0, :]).squeeze(0)                   # (D,)

# --------------------- TransMIL-Temporal (video) -----------------------
class TPEG(nn.Module):
    """
    Temporal Positional Encoding Generator (1-D):
    depthwise Conv1d with 3/5/7 kernels over the frame sequence, then fuse.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dw3  = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dw5  = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.dw7  = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.proj = nn.Linear(3 * dim, dim)
        self.act  = nn.GELU()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x  = tokens.transpose(0, 1).unsqueeze(0)          # (1,D,T)
        y3 = self.dw3(x); y5 = self.dw5(x); y7 = self.dw7(x)
        y  = torch.cat([y3, y5, y7], dim=1)               # (1,3D,T)
        y  = y.squeeze(0).transpose(0, 1)                 # (T,3D)
        return self.act(self.proj(y))                     # (T,D)

def _make_banded_attn_mask(seq_len: int, window: int, device, allow_cls: bool=True):
    """
    Additive attention mask for TransformerEncoder (float):
      mask[i,j] = 0 (allow), -inf (block).
    Shape: (S,S), S = 1 + seq_len (includes CLS at idx 0).
    """
    if window <= 0:
        return None
    S = 1 + seq_len
    mask = torch.zeros((S, S), dtype=torch.float32, device=device)
    # block content tokens outside band
    idx = torch.arange(S, device=device)
    ii = idx[1:].unsqueeze(1)         # (T,1)
    jj = idx[1:].unsqueeze(0)         # (1,T)
    outside = (jj - ii).abs() > window
    mask[1:, 1:][outside] = float("-inf")
    if allow_cls:
        mask[0, :] = 0.0              # CLS attends to all
        mask[:, 0] = 0.0              # all attend to CLS
    mask.fill_diagonal_(0.0)          # never block self
    return mask

class TransNILTemporalPool(nn.Module):
    """
    TransMIL-style temporal pooler (frames as instances):
      - residual TPEG (learned, multi-scale temporal PE)
      - optional temporal downsampling (stride s)
      - optional windowed attention (bandwidth 'window')
      - [CLS] token + L× TransformerEncoder
      - returns normalized [CLS] as pooled feature (D,)
    """
    def __init__(self, dim: int, heads: int = 8, depth: int = 3,
                 ff_mult: int = 4, dropout: float = 0.1,
                 use_tpeg: bool = True, window: int = 64, stride: int = 1):
        super().__init__()
        self.use_tpeg = use_tpeg
        self.window   = int(window)
        self.stride   = int(max(1, stride))

        # Optional temporal downsampler (depthwise + pointwise)
        if self.stride > 1:
            self.down = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, stride=self.stride, groups=dim),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=1),
            )
        else:
            self.down = None

        self.tpeg = TPEG(dim) if use_tpeg else None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff_mult * dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (T, D)
        device = x.device
        if self.down is not None:
            xd = self.down(x.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        else:
            xd = x

        if self.tpeg is not None:
            xd = xd + self.tpeg(xd)                      # residual TPEG

        Tprime = xd.size(0)
        xs = xd.unsqueeze(0)                             # (1, T', D)
        cls = self.cls_token.expand(1, -1, -1).to(device)  # (1,1,D)
        src = torch.cat([cls, xs], dim=1)                # (1, 1+T', D)

        attn_mask = _make_banded_attn_mask(Tprime, self.window, device, allow_cls=True)
        out = self.encoder(src, mask=attn_mask)          # (1, 1+T', D)

        return self.norm_out(out[:, 0, :]).squeeze(0)    # (D,)

# ----------------------------- factory ---------------------------------
def build_pooler(name: str, dim: int):
    """
    Recognized names:
      - "mean", "max", "attention", "gated", "abmil", "abmil{E}_nogate", "abmil_tpeg", "mh_gated{K}", "temporal_conv{out}_{heads}", "gru", "transf{H}"
      - 2-D: "transmil" or "transnil" (optionally "..._noppeg")
      - 1-D (video): "transmil1d{heads}_{layers}_w{win}_s{stride}" (aliases: transnil1d, transmil_temporal, transnil_temporal)
        Examples:
          "transmil1d" (defaults 8 heads, 3 layers, w64, s1)
          "transmil1d8_3_w64_s2"
          "transmil1d8_3" (global attention if no wNN given)
          "transmil1d8_3_notpeg"
    """
    name = name.lower()

    if name == "mean":        return MeanPool()
    if name == "max":         return MaxPool()
    if name == "attention":   return AttentionMIL(dim)
    if name == "gated":       return GatedAttentionMIL(dim)

    # ------------ ABMIL (MIL-Lab) ------------------
    # "abmil" (gated, default dims), "abmil_nogate", "abmil512", "abmil256_nogate"
    if name.startswith("abmil") and not name.startswith("abmil_t"):
        gate = "nogate" not in name
        nums = list(map(int, re.findall(r"\d+", name)))
        embed = nums[0] if nums else 512
        return ABMILPool(dim, embed_dim=embed, gate=gate)

    # ------------ Temporal ABMIL (ABMIL + TPEG) ---
    # "abmil_tpeg" (gated), "abmil_tpeg_nogate"
    if name.startswith("abmil_t"):
        gate = "nogate" not in name
        nums = list(map(int, re.findall(r"\d+", name)))
        embed = nums[0] if nums else 512
        return TemporalABMILPool(dim, embed_dim=embed, gate=gate)

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

    # ------------ optional extras ------------------
    if name == "gru":         return GRUPool(dim)
    if name.startswith("transf"):
        h = int((re.findall(r"\d+", name) or ["4"])[0])
        return TinyTransformerPool(dim, heads=h)

    # ------------ TransMIL / TransNIL (2-D grid) ---
    if name.startswith("transmil") and not name.startswith("transmil1d"):
        nums = list(map(int, re.findall(r"\d+", name)))
        heads = nums[0] if len(nums) > 0 else 4
        depth = nums[1] if len(nums) > 1 else 2
        use_ppeg = ("noppeg" not in name)
        return TransNILPool(dim, heads=heads, depth=depth, use_ppeg=use_ppeg)
    if name.startswith("transnil") and not name.startswith("transnil1d"):
        nums = list(map(int, re.findall(r"\d+", name)))
        heads = nums[0] if len(nums) > 0 else 4
        depth = nums[1] if len(nums) > 1 else 2
        use_ppeg = ("noppeg" not in name)
        return TransNILPool(dim, heads=heads, depth=depth, use_ppeg=use_ppeg)

    # ------ TransMIL-Temporal / TransNIL-Temporal ---
    if (name.startswith("transmil1d") or name.startswith("transnil1d")
        or name.startswith("transmil_temporal") or name.startswith("transnil_temporal")):
        nums   = list(map(int, re.findall(r"\d+", name)))     # heads, depth, (optional) window, stride
        heads  = nums[0] if len(nums) > 0 else 8
        depth  = nums[1] if len(nums) > 1 else 3
        w_match = re.search(r"w(\d+)", name)
        s_match = re.search(r"s(\d+)", name)
        window  = int(w_match.group(1)) if w_match else (nums[2] if len(nums) > 2 else 64)
        stride  = int(s_match.group(1)) if s_match else (nums[3] if len(nums) > 3 else 1)
        use_tp  = ("notpeg" not in name)
        if not w_match and len(nums) < 3:
            window = 0  # no explicit window → global attention
        return TransNILTemporalPool(dim, heads=heads, depth=depth,
                                    use_tpeg=use_tp, window=window, stride=stride)

    raise ValueError(f"Unknown aggregator '{name}'")

# --------------------------- main model --------------------------------
class MILModel(nn.Module):
    """CNN/ViT backbone ➜ MIL pooler ➜ regressor (no chunking)."""
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
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
        self.regressor = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, bag: torch.Tensor):   # bag: (T, C, H, W)
        feats = self.backbone(bag)          # (T, D)
        pooled = self.aggregator(feats)     # (D,)
        return self.regressor(pooled)       # (1,)
