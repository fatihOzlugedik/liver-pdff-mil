# Liver PDFF MIL

Multiple Instance Learning (MIL) for PDFF (Proton Density Fat Fraction) regression from ultrasound videos. Supports multi-task learning with a joint regression + classification head, multiple loss functions, and YAML-driven configuration.

---

## Quick Start

```bash
# 1. Copy the template and fill in your data paths
cp configs/default.yaml configs/my_experiment.yaml

# 2. Edit paths and settings
nano configs/my_experiment.yaml

# 3. Run
python src/main.py configs/my_experiment.yaml
```

---

## Project Structure

```
liver-pdff-mil/
├── configs/                 # ← ONLY EDIT THESE
│   ├── default.yaml         # Full template with all options documented
│   ├── ablation/            # 64 pre-made configs (8 aggregators × 8 losses)
│   └── *.yaml               # Example configs
│
├── src/                     # Source code (do not edit)
│   ├── main.py              # Entry point
│   ├── model.py             # MIL model + pooling strategies
│   ├── engine.py            # Training/validation loop
│   ├── losses.py            # Loss functions
│   ├── data.py              # Dataset and data loading
│   └── config.py            # YAML loader (internal)
│
├── run_ablation.sh          # Multi-GPU scheduler for ablation studies
└── README.md
```

---

## Configuration

All settings live in a single YAML file. Copy `configs/default.yaml` and modify — **you never need to edit any `.py` file**.

```bash
python src/main.py configs/your_config.yaml
```

### Minimal config

```yaml
# Data
video_folder: "/path/to/videos"
csv_path: "/path/to/dataset.csv"
cache_dir: "/path/to/frame_cache"
output_dir: "/path/to/runs"

# Model
aggregator: "abmil"

# Loss
loss_fn: "l1"

# Training
epochs: 100
n_folds: 5
```

---

## Aggregators (MIL Pooling)

Set with `aggregator:` in the config.

| Name | Description |
|------|-------------|
| `mean` | Simple mean pooling |
| `max` | Max pooling |
| `attention` | Single-head additive attention |
| `gated` | Single-head gated attention |
| `abmil` | Attention-based MIL — recommended default |
| `mh_gated4` | Multi-head gated attention (4 heads) |
| `mh_gated8` | Multi-head gated attention (8 heads) |
| `temporal_conv32h4` | Temporal conv + gated attention |

---

## Loss Functions

Set with `loss_fn:` and optional `loss_config:` in the config.

| Name | Description | Key config params |
|------|-------------|-------------------|
| `l1` | MAE — robust default | — |
| `l2` / `mse` | MSE | — |
| `huber` | Huber — good middle ground | `delta: 2.0` |
| `logcosh` | Smooth, outlier-tolerant | — |
| `wing` | Wing loss — face/landmark style | `w: 5.0, epsilon: 2.0` |
| `focal` | Focus on hard examples | `gamma: 2.0, base_loss: "l1"` |
| `weighted_zone` | Higher penalty in clinical zones | `base_loss: "l1"` |
| `threshold_aware` | Penalty for crossing PDFF thresholds | `thresholds: [5.0], crossing_penalty: 1.0` |
| `combined` | Weighted sum of multiple losses | see example below |

### Examples

```yaml
# Huber
loss_fn: "huber"
loss_config:
  delta: 2.0

# Focal
loss_fn: "focal"
loss_config:
  gamma: 2.0
  base_loss: "l1"

# Threshold-aware (penalizes crossing clinical cutoffs)
loss_fn: "threshold_aware"
loss_config:
  thresholds: [6.4, 16.3, 20.7]
  crossing_penalty: 2.0

# Combined
loss_fn: "combined"
loss_config:
  losses:
    l1: 0.5
    huber: 0.3
    logcosh: 0.2
  loss_configs:
    huber:
      delta: 2.0
```

### Presets

Presets are named bundles of `loss_fn` + `loss_config`. They override `loss_fn` when set.

```yaml
loss_preset: "clinical"      # Threshold-aware at PDFF clinical cutoffs
# loss_preset: "robust"      # Huber delta=2.0
# loss_preset: "focal_l1"    # Focal gamma=2.0 with L1 base
# loss_preset: "zone_weighted"
```

---

## Classification Head (Multi-task)

The model can simultaneously predict PDFF value (regression) and PDFF stage (classification). Classification targets are derived automatically from the regression target using configurable thresholds.

### PDFF Staging (default thresholds)

| Class | Stage | PDFF |
|-------|-------|------|
| 0 | Normal | < 6.4% |
| 1 | Mild steatosis | 6.4% – 16.3% |
| 2 | Moderate steatosis | 16.3% – 20.7% |
| 3 | Severe steatosis | > 20.7% |

### Config

```yaml
use_classifier: true         # Enable classification head
cls_weight: 0.3              # Weight on classification loss (regression = 1 - cls_weight)
num_classes: 4
cls_thresholds: [6.4, 16.3, 20.7]
```

`cls_weight` ranges from 0 (regression only) to 1 (classification only). Use `use_classifier: false` to disable the head entirely.

---

## Cross-Validation

5-fold CV is the default. Fold assignments must be in the CSV as columns named `set1`, `set2`, ..., `set5`.

```yaml
n_folds: 5
fold_set_prefix: "set"    # Reads columns: set1, set2, ..., set5
```

To run a single split instead:

```yaml
n_folds: 0
set_col: "set1"           # Column containing train/val/test labels
```

---

## Output

Each run creates a directory under `output_dir/`:

```
runs/abmil_focal_cls30_CV5/
├── config.json              # Saved config snapshot
├── train_log.csv            # Per-epoch metrics
├── train_log.txt            # Full training log
├── train_log.png            # Loss/metric curves
├── best.pt                  # Best model checkpoint
├── fold00/
│   ├── val_predictions.csv
│   ├── test_predictions.csv
│   ├── val_summary.png
│   └── test_summary.png
├── fold01/
└── ...
```

---

## Ablation Study (8 × 8 = 64 experiments)

Pre-made configs covering all combinations of aggregators and losses are in `configs/ablation/`. Each is configured with `use_classifier=true`, `cls_weight=0.3`, and `n_folds=5`.

### Run on a single GPU

```bash
python src/main.py configs/ablation/abmil_focal.yaml
```

### Run all 64 experiments across 4 GPUs

```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

The scheduler polls every 60 seconds and launches new jobs as GPUs become free. Logs go to `logs/ablation/<config_name>.log`.

Config naming convention: `<aggregator>_<loss>.yaml`

```
mean_l1.yaml, mean_l2.yaml, mean_huber.yaml, ...
abmil_l1.yaml, abmil_focal.yaml, abmil_threshold_aware.yaml, ...
temporal_conv32h4_wing.yaml, ...
```

---

## Frame Cache

Pre-compute and cache decoded frames for faster training:

```bash
python src/precompute_cache.py --img_size 384
```

Set `cache_dir` in your config to point to the output directory.

---

## Requirements

```
torch
torchvision
timm
pandas
numpy
scikit-learn
pyyaml
tqdm
matplotlib
decord         # optional — faster video decoding
```

---

## Tips

- **Start here**: `configs/default.yaml` has every option documented inline
- **Aggregator**: `abmil` is a strong default; try `temporal_conv32h4` if temporal ordering matters
- **Loss**: Start with `l1`, then try `huber` or `threshold_aware` for clinical tasks
- **Classification weight**: 0.3 works well; increase if staging accuracy is the primary metric
- **Check logs**: `tail -f logs/ablation/<run>.log` to monitor ablation jobs
