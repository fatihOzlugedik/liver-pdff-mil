# data.py
"""
Key design decisions:
  - RAM cache: all train+val videos loaded into RAM at dataset init.
    Since all 4 parallel folds read from the same .npy disk cache,
    disk I/O is already fast — the RAM cache eliminates even that per-epoch.
    4 parallel folds = 4x RAM usage, but each fold only loads its own
    train+val split (NOT test), so actual RAM per fold is ~20-25 GB
    rather than the full 33 GB.
  - Test set is loaded LAZILY — only when evaluate_val_test() is called,
    not at training start. This avoids holding test frames in RAM during
    the entire training run. At evaluation time, test videos are loaded
    once into RAM for fast inference.
  - Frame sampling: train = random (different each epoch, but deterministic
    via worker seed), val/test = fixed equally-spaced indices.
  - Worker seeding: each DataLoader worker is seeded with (base_seed + worker_id)
    so frame sampling is fully reproducible across runs, workers, and aggregators.
"""

import random
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from config import CFG
import numpy as np

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    from torchvision.io import read_video
    print("[WARN] decord not found, falling back to torchvision.io.read_video (slower)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_sec(val) -> float:
    """Convert MM.SS float format to total seconds. E.g. 1.30 → 90.0"""
    val     = float(val)
    minutes = int(val)
    seconds = round((val - minutes) * 100)
    return float(minutes * 60 + seconds)


def _load_video_frames(vid_path: str, start_sec: float, end_sec: float) -> np.ndarray:
    """Load frames from [start_sec, end_sec]. Returns (T, H, W, 3) uint8."""
    if DECORD_AVAILABLE:
        vr      = VideoReader(vid_path, ctx=decord_cpu(0))
        fps     = vr.get_avg_fps()
        total   = len(vr)
        start_f = max(0, int(start_sec * fps))
        end_f   = min(total - 1, int(end_sec * fps))
        if end_f <= start_f:
            end_f = min(start_f + 1, total - 1)
        return vr.get_batch(list(range(start_f, end_f + 1))).asnumpy()
    else:
        frames, _, _ = read_video(
            vid_path, start_pts=start_sec, end_pts=end_sec, pts_unit="sec"
        )
        return frames.numpy()


def _load_frames_for_row(row, cfg: CFG) -> np.ndarray:
    """Load frames for a single CSV row, using disk cache if available."""
    vid      = str(row[cfg.video_id_col])
    vid_path = str(row[cfg.video_path_col])
    start_t  = _to_sec(row[cfg.start_col])
    end_t    = _to_sec(row[cfg.end_col])
    if end_t < start_t:
        start_t, end_t = end_t, start_t

    cache_path = cfg.cache_dir / f"{vid}.npy"
    if cache_path.exists():
        return np.load(str(cache_path))
    return _load_video_frames(vid_path, start_t, end_t)


def worker_init_fn(worker_id: int):
    """
    Seed each DataLoader worker independently but deterministically.
    Called once per worker at spawn time.
    The base seed is passed via worker_init_fn closure (set in _mk_loader).
    This ensures frame sampling is reproducible across runs while each
    worker has a unique but fixed random state.
    """
    seed = torch.initial_seed() % (2**32)  # derived from main seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LiverPDFFDataset(Dataset):
    """
    One row = one video = one MIL bag.

    RAM cache strategy:
      - eager=True  (train/val): all videos loaded into RAM at init.
        Eliminates disk I/O during training epochs.
      - eager=False (test): videos loaded from disk cache on __getitem__.
        Avoids occupying RAM with test frames during the full training run.
        At evaluation time, caller can optionally pre-load into RAM by
        calling load_into_ram() explicitly.

    Frame sampling:
      - train: random n_frames indices per epoch (reproducible via worker seed)
      - val/test: fixed equally-spaced n_frames indices (always deterministic)
    """

    def __init__(self,
                 df: pd.DataFrame,
                 cfg: CFG,
                 split: str,
                 augment: bool = False,
                 eager: bool = True):
        self.df       = df.reset_index(drop=True)
        self.cfg      = cfg
        self.split    = split
        self.n_frames = int(getattr(cfg, "n_frames", 600))
        self.eager    = eager

        size     = int(getattr(cfg, 'img_size', 224))
        self.aug = SafeAugmentation(size) if augment else T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])

        self.ram_cache: dict[str, np.ndarray | None] = {}

        if eager:
            self._load_all_into_ram()
        else:
            print(f"[CACHE] {split}: lazy mode — {len(self.df)} videos (loaded on demand).",
                  flush=True)
            self._validate_lazy()

    # ------------------------------------------------------------------
    def _load_all_into_ram(self):
        """Load all videos into RAM. Called once at dataset init for train/val."""
        print(f"[CACHE] {self.split}: loading {len(self.df)} videos into RAM...",
              flush=True)
        valid_vids = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            vid = str(row[self.cfg.video_id_col])
            try:
                frames = _load_frames_for_row(row, self.cfg)
                if frames.shape[0] == 0:
                    raise RuntimeError("0 frames returned")
                self.ram_cache[vid] = frames
                valid_vids.append(vid)
                print(f"  [{i+1}/{len(self.df)}] {vid}: {frames.shape[0]} frames", flush=True)
            except Exception as exc:
                print(f"  [{i+1}/{len(self.df)}] [SKIP] {vid}: {exc}", flush=True)
                self.ram_cache[vid] = None

        n_ok   = len(valid_vids)
        n_skip = len(self.df) - n_ok
        print(f"[CACHE] {self.split} ready: {n_ok} OK, {n_skip} skipped.\n", flush=True)

        self.df = self.df[
            self.df[self.cfg.video_id_col].astype(str).isin(valid_vids)
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    def _validate_lazy(self):
        """Check which videos are accessible without loading into RAM."""
        import os
        valid_vids = []
        for i in range(len(self.df)):
            row        = self.df.iloc[i]
            vid        = str(row[self.cfg.video_id_col])
            cache_path = self.cfg.cache_dir / f"{vid}.npy"
            vid_path   = str(row[self.cfg.video_path_col])
            if cache_path.exists() or os.path.exists(vid_path):
                valid_vids.append(vid)
            else:
                print(f"  [SKIP] {vid}: neither cache nor video file found", flush=True)

        self.df = self.df[
            self.df[self.cfg.video_id_col].astype(str).isin(valid_vids)
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    def load_into_ram(self):
        """
        Explicitly load all lazy videos into RAM.
        Call this before evaluation to speed up inference.
        """
        if self.eager:
            return  # already in RAM
        print(f"[CACHE] {self.split}: loading into RAM for evaluation...", flush=True)
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            vid = str(row[self.cfg.video_id_col])
            if vid not in self.ram_cache:
                try:
                    self.ram_cache[vid] = _load_frames_for_row(row, self.cfg)
                except Exception as exc:
                    print(f"  [WARN] {vid}: {exc}", flush=True)
                    self.ram_cache[vid] = None
        self.eager = True  # switch to RAM mode
        print(f"[CACHE] {self.split} loaded into RAM.\n", flush=True)

    # ------------------------------------------------------------------
    def _get_frames(self, row) -> np.ndarray:
        """Return raw frames — from RAM cache or disk."""
        vid = str(row[self.cfg.video_id_col])
        if self.eager and vid in self.ram_cache and self.ram_cache[vid] is not None:
            return self.ram_cache[vid]
        return _load_frames_for_row(row, self.cfg)

    # ------------------------------------------------------------------
    # OLD: Pure random sampling — picks n_frames indices uniformly at random.
    # Can over-represent some temporal regions and skip others entirely.
    # def _sample_indices(self, T: int) -> np.ndarray:
    #     if self.split == "train":
    #         # np.random is seeded per-worker via worker_init_fn → reproducible
    #         idx = np.random.choice(T, size=self.n_frames, replace=(T < self.n_frames))
    #         idx.sort()
    #         return idx
    #     else:
    #         return np.linspace(0, T - 1, num=self.n_frames, dtype=int)

    # NEW: Stratified temporal sampling — divides the video into n_frames
    # equal-length bins and picks one random frame per bin. This guarantees
    # uniform temporal coverage every epoch (no segments are skipped) while
    # still being stochastic (different frame within each bin each epoch).
    # Reduces gradient variance and ensures diagnostically relevant segments
    # are always represented in the MIL bag.
    def _sample_indices(self, T: int) -> np.ndarray:
        if self.split == "train":
            n = self.n_frames
            if T >= n:
                # Split [0, T) into n equal bins; sample one frame per bin
                edges = np.linspace(0, T, num=n + 1, dtype=np.float64)
                idx = np.array([
                    np.random.randint(int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1])))
                    for i in range(n)
                ], dtype=int)
            else:
                # Fewer frames than requested: keep all T frames,
                # fill remaining slots by random resampling with replacement
                base = np.arange(T)
                extra = np.random.choice(T, size=n - T, replace=True)
                idx = np.concatenate([base, extra])
                idx.sort()
            return idx
        else:
            return np.linspace(0, T - 1, num=self.n_frames, dtype=int)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.df)

    def _pdff_to_class(self, pdff: float) -> int:
        """
        Convert PDFF percentage to class index based on thresholds.

        Default thresholds (from config):
          Class 0: Normal (< 6.4%)
          Class 1: Mild steatosis (6.4% - 16.3%)
          Class 2: Moderate steatosis (16.3% - 20.7%)
          Class 3: Severe steatosis (> 20.7%)
        """
        thresholds = getattr(self.cfg, "cls_thresholds", [6.4, 16.3, 20.7])
        for i, thresh in enumerate(thresholds):
            if pdff < thresh:
                return i
        return len(thresholds)  # Last class

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        vid   = str(row[self.cfg.video_id_col])
        pdff  = float(row[self.cfg.target_col])
        label = torch.tensor(pdff, dtype=torch.float32)

        # Get class label for classification head
        cls_label = torch.tensor(self._pdff_to_class(pdff), dtype=torch.long)

        frames  = self._get_frames(row)
        T_total = frames.shape[0]
        idxs    = self._sample_indices(T_total)
        frames  = frames[idxs]                   # (n_frames, H, W, 3)

        bag = []
        for fr in frames:
            # Cache stores (224, 224) uint8 grayscale — no convert needed.
            # Fallback: if raw RGB (H, W, 3) frame, convert to grayscale.
            if fr.ndim == 3:
                pil = Image.fromarray(fr).convert("L")
            else:
                pil = Image.fromarray(fr, mode="L")
            bag.append(self.aug(pil))             # (1, H, W)
        bag = torch.stack(bag)                    # (n_frames, 1, H, W)

        return bag, label, cls_label, vid


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class SafeAugmentation(T.Compose):
    def __init__(self, size: int = 224):
        super().__init__([
            T.Lambda(
                lambda img: TF.rotate(img, angle=random.uniform(-20, 20),
                                      center=(img.width // 2, 0))
                if random.random() < 0.8 else img
            ),
            T.RandomHorizontalFlip(0.3),
            T.RandomApply([T.GaussianBlur(3, (0.1, 1.5))], p=0.5),
            T.Resize((size, size)),
            T.ToTensor(),
            AddSpeckleNoise(0.1),
        ])


class AddSpeckleNoise:
    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t + t * torch.randn_like(t) * self.std


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def _collate(batch):
    bag, y, cls_y, pid = zip(*batch)
    return bag[0], y[0], cls_y[0], pid[0]


def _mk_loader(ds: Dataset, shuffle: bool, cfg: CFG, sampler=None) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        collate_fn=_collate,
        num_workers=cfg.workers,
        worker_init_fn=worker_init_fn,   # deterministic per-worker seeding
        persistent_workers=(cfg.workers > 0),
    )


def _make_weighted_sampler(df: pd.DataFrame, cfg: CFG):
    if not bool(getattr(cfg, "use_weighted_sampler", False)):
        return None

    balance_col = getattr(cfg, "balance_col", None) or getattr(cfg, "stratify_col", None)
    if balance_col is None or balance_col not in df.columns:
        return None

    labels = df[balance_col].astype("category").cat.codes.to_numpy()
    if labels.size == 0:
        return None

    counts = np.bincount(labels)
    counts[counts == 0] = 1
    sample_weights = (1.0 / counts)[labels]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_loaders(cfg: CFG, fold_idx: int | None = None):
    """
    Build train/val loaders with eager RAM cache.
    Test dataset is lazy — loaded on demand at evaluation time.

    Returns: tr_dl, val_dl, tst_dl, tr_df, val_df, tst_df
    """
    df = pd.read_csv(cfg.csv_path)

    if cfg.n_folds and cfg.n_folds > 0:
        assert fold_idx is not None, "fold_idx must be provided when n_folds > 0"
        set_col = f"{getattr(cfg, 'fold_set_prefix', 'set')}{fold_idx + 1}"
    else:
        set_col = cfg.set_col

    if set_col not in df.columns:
        raise KeyError(f"Split column '{set_col}' not found. Available: {list(df.columns)}")

    df = df[df[set_col].notna()].reset_index(drop=True)

    tr_df  = df[df[set_col] == "train"].reset_index(drop=True)
    val_df = df[df[set_col] == "val"  ].reset_index(drop=True)
    tst_df = df[df[set_col] == "test" ].reset_index(drop=True)

    # Train & val: eager RAM cache (used every epoch)
    ds_tr  = LiverPDFFDataset(tr_df,  cfg, "train", augment=True,  eager=True)
    ds_val = LiverPDFFDataset(val_df, cfg, "val",   augment=False, eager=True)

    # Test: lazy — RAM loaded only at evaluation time via load_into_ram()
    ds_tst = LiverPDFFDataset(tst_df, cfg, "test",  augment=False, eager=False)

    # Build sampler AFTER dataset init — ds_tr.df may be smaller if videos were skipped
    tr_sampler = _make_weighted_sampler(ds_tr.df, cfg)

    return (
        _mk_loader(ds_tr,  True,  cfg, sampler=tr_sampler),
        _mk_loader(ds_val, False, cfg),
        _mk_loader(ds_tst, False, cfg),
        tr_df, val_df, tst_df,
    )