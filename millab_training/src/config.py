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
    video_folder : Path = Path("/research/projects/Sahika/projects/liver_PDFF/data/EUS_videos_v5_feb26")
    csv_path     : Path = Path("/research/projects/Sahika/projects/liver_PDFF/data/dataset_v5_5fold_feb26_trainval_only.csv")

    set_col      : str  = "set1"
    fold_set_prefix: str = "set"

    video_id_col   : str = "video"
    video_path_col : str = "video_path"
    target_col     : str = "PDFF"
    start_col      : str = "start_time_1"
    end_col        : str = "end_time_1"

    img_size       : int = 224
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
    window_size : int = 0
    aggregator : str = "abmil"
    embed_dim  : int = 512
    run_name   : Optional[str] = None

    # ---------------- k-fold ---------------
    n_folds      : int  = 0
    stratify_col : str  = "stage"

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
        ws_tag = "_ws" if self.use_weighted_sampler else ""
        cv_tag = f"_CV{self.n_folds}" if self.n_folds > 0 else ""
        size_tag = f"_s{self.img_size}"
        tag = self.run_name or f"{self.backbone}/{self.aggregator}/{self.video_folder.name}_{self.epochs}ep{ws_tag}{cv_tag}"
        if not tag.endswith(f"_s{self.img_size}"):
            tag = tag + size_tag
        if f"_f{self.n_frames}" not in tag:
            tag = tag + f"_f{self.n_frames}"

        self.cache_dir = Path(f"/research/projects/Sahika/projects/liver_PDFF/data/frame_cache_{self.img_size}")

        self.root_dir = Path(f"/research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/millab_training/runs/{tag}")
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
