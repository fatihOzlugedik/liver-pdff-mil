#!/usr/bin/env python3
"""
Generate all ablation config files for aggregator × loss combinations.
"""

import yaml
from pathlib import Path

# Output directory
CONFIG_DIR = Path("configs/ablation")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Base configuration (edit paths as needed)
BASE = {
    "video_folder": "/research/projects/Sahika/projects/liver_PDFF/data/EUS_videos_v5_feb26",
    "csv_path": "/research/projects/Sahika/projects/liver_PDFF/data/dataset_v5_5fold_feb26_trainval_only.csv",
    "cache_dir": "/research/projects/Sahika/projects/liver_PDFF/data/frame_cache_384",
    "output_dir": "/research/projects/Sahika/projects/liver_PDFF/runs/ablation",

    "video_id_col": "video",
    "video_path_col": "video_path",
    "target_col": "PDFF",
    "start_col": "start_time_1",
    "end_col": "end_time_1",
    "stratify_col": "stage",

    "img_size": 384,
    "n_frames": 175,
    "backbone": "swin_tiny_patch4_window7_224.ms_in1k",
    "window_size": 0,

    # Classification head - always enabled
    "use_classifier": True,
    "cls_weight": 0.3,
    "num_classes": 4,
    "cls_thresholds": [6.4, 16.3, 20.7],

    # Training
    "epochs": 100,
    "batch_size": 1,
    "grad_accum": 8,
    "lr": 0.0001,
    "wd": 0.0001,
    "workers": 2,
    "seed": 42,
    "threshold": 5.0,
    "use_weighted_sampler": True,

    # CV
    "n_folds": 5,
    "fold_set_prefix": "set",
    "set_col": "set1",
}

# All aggregators
AGGREGATORS = [
    "mean",
    "max",
    "attention",
    "gated",
    "abmil",
    "mh_gated4",
    "mh_gated8",
    "temporal_conv32h4",
]

# All loss configurations with meaningful parameters
LOSSES = {
    "l1": {
        "loss_fn": "l1",
        "loss_config": {},
    },
    "l2": {
        "loss_fn": "l2",
        "loss_config": {},
    },
    "huber": {
        "loss_fn": "huber",
        "loss_config": {"delta": 2.0},
    },
    "logcosh": {
        "loss_fn": "logcosh",
        "loss_config": {},
    },
    "wing": {
        "loss_fn": "wing",
        "loss_config": {"w": 5.0, "epsilon": 2.0},
    },
    "focal": {
        "loss_fn": "focal",
        "loss_config": {"gamma": 2.0, "base_loss": "l1"},
    },
    "weighted_zone": {
        "loss_fn": "weighted_zone",
        "loss_config": {"base_loss": "l1"},
    },
    "threshold_aware": {
        "loss_fn": "threshold_aware",
        "loss_config": {
            "base_loss": "l1",
            "thresholds": [5.0, 6.4, 16.3, 20.7],
            "crossing_penalty": 1.0,
        },
    },
}

def generate_configs():
    """Generate all config files."""
    configs = []

    for agg in AGGREGATORS:
        for loss_name, loss_cfg in LOSSES.items():
            # Config name
            name = f"{agg}_{loss_name}"

            # Merge configs
            cfg = {**BASE}
            cfg["aggregator"] = agg
            cfg["loss_fn"] = loss_cfg["loss_fn"]
            cfg["loss_config"] = loss_cfg["loss_config"]
            cfg["run_name"] = name

            # Save config
            cfg_path = CONFIG_DIR / f"{name}.yaml"
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

            configs.append(name)
            print(f"Created: {cfg_path}")

    print(f"\nTotal configs: {len(configs)}")

    # Save list of all configs
    with open(CONFIG_DIR / "all_configs.txt", "w") as f:
        for name in configs:
            f.write(f"{name}\n")

    return configs

if __name__ == "__main__":
    generate_configs()
