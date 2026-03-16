# config.py
from dataclasses import dataclass, field, asdict
from pathlib import Path
import torch, json
import numpy as np
import random
from typing import Optional

@dataclass
class CFG:
    # ---------------- data ----------------
    # Folder that contains the original mp4 files.
    video_folder : Path = Path("/research/projects/Sahika/projects/liver_PDFF/data/EUS_videos_v5_feb26")
    # CSV that includes columns like: video_path, PDFF, start_time_1, end_time_1, set1..set5
    csv_path     : Path = Path("/research/projects/Sahika/projects/liver_PDFF/data/dataset_v5_5fold_feb26_trainval_only.csv")

    # One-run: column that contains split labels train/val/test.
    # For k-fold with precomputed columns, we will use set{fold} (e.g., set1..set5).
    set_col      : str  = "set1"
    fold_set_prefix: str = "set"  # set1..set5

    # Column mapping for your dataframe
    video_id_col   : str = "video"       # e.g., EUS01, EUSAI_241
    video_path_col : str = "video_path"  # full path to .mp4
    target_col     : str = "PDFF"        # regression target
    start_col      : str = "start_time_1"
    end_col        : str = "end_time_1"

    # Frame size — used by precompute_cache.py and data.py
    # Cache is stored per-size: frame_cache_224/, frame_cache_384/, etc.
    img_size       : int = 224

    # Frame sampling
    n_frames       : int = 600

    # -------------- training --------------
    batch_size : int   = 1
    grad_accum : int   = 8
    epochs     : int   = 100
    lr         : float = 1e-4
    wd         : float = 1e-4
    workers    : int   = 2
    threshold  : float = 5.0

    # ---------------- model ----------------
    backbone    : str = "swin_tiny_patch4_window7_224.ms_in1k"
    window_size : int = 0   # 0 = use model default; set to 12 for swin 384
    # For video: TransMIL-Temporal with CLS pooling (works with your model.py)
    aggregator : str = "mean" #overrided
    run_name   : Optional[str] = None  # can override folder naming from a script

    # ---------------- k-fold ---------------
    n_folds      : int  = 0
    stratify_col : str  = "stage"  # used only if you re-create folds on the fly

    # --------------- runtime ---------------
    device : str = "cuda" if torch.cuda.is_available() else "cpu"
    seed   : int = 42

    # ---------- weighted sampling ----------
    use_weighted_sampler: bool = True

    # --------------- derived ---------------
    root_dir     : Path = field(init=False)
    exp_dir      : Path = field(init=False)
    log_file     : Path = field(init=False)
    log_csv_path : Path = field(init=False)

    def __post_init__(self):
        # Build a sensible tag: backbone/aggregator/videoFolder_epochs[_ws]
        ws_tag = "_ws" if self.use_weighted_sampler else ""
        cv_tag = f"_CV{self.n_folds}" if self.n_folds > 0 else ""
        size_tag = f"_s{self.img_size}"
        tag = self.run_name or f"{self.backbone}/{self.aggregator}/{self.video_folder.name}_{self.epochs}ep{ws_tag}{cv_tag}"
        # Append size tag only if not already present (avoids double-suffix when run_name
        # is supplied pre-suffixed by run_all.py / ablation_run_all.py)
        if not tag.endswith(f"_s{self.img_size}"):
            tag = tag + size_tag
        if f"_f{self.n_frames}" not in tag:
            tag = tag + f"_f{self.n_frames}"

        # Cache dir is size-specific so different img_size runs don't collide
        self.cache_dir = Path(f"/research/projects/Sahika/projects/liver_PDFF/data/frame_cache_{self.img_size}")

        self.root_dir = Path(f"/research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/runs_without_test/{tag}")
        self.exp_dir = self.root_dir
        self.log_file = self.exp_dir / "training.log"
        self.log_csv_path = self.exp_dir / "cv_folds.csv"

        print(f"[INFO] Saving to {self.exp_dir}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # deterministic GPU ops
    torch.backends.cudnn.benchmark     = False  # no auto-tuning (changes op selection)