# Liver PDFF MIL

Multiple Instance Learning (MIL) for PDFF (Proton Density Fat Fraction) regression from ultrasound videos.

## Quick Start

```bash
# 1. Copy and edit config
cp configs/default.yaml configs/my_experiment.yaml
nano configs/my_experiment.yaml

# 2. Run
python src/main.py configs/my_experiment.yaml
```

---

## Project Structure

```
liver-pdff-mil/
├── configs/                 # ← EDIT THESE YAML FILES
│   ├── default.yaml         # Full template (start here)
│   ├── abmil_huber.yaml
│   ├── mean_focal.yaml
│   └── ...
│
├── src/                     # Source code (don't edit)
│   ├── main.py              # Entry point
│   ├── model.py             # MIL model architecture
│   ├── engine.py            # Training loop
│   ├── losses.py            # Loss functions
│   ├── data.py              # Data loading
│   └── config.py            # Config loader (internal)
│
└── README.md
```

---

## Configuration

All settings are controlled through YAML files. Copy `configs/default.yaml` and modify as needed.

### Key Settings

```yaml
# === MODEL ===
aggregator: "abmil"          # MIL pooling method
backbone: "swin_tiny_patch4_window7_224.ms_in1k"

# === LOSS ===
loss_fn: "l1"                # Loss function
loss_config: {}              # Loss parameters

# === MULTI-TASK ===
use_classifier: true         # Enable classification head
cls_weight: 0.3              # Classification weight (0-1)

# === TRAINING ===
epochs: 100
n_folds: 5                   # Cross-validation folds
```

---

## Available Options

### Aggregators (MIL Pooling)

| Name | Description |
|------|-------------|
| `mean` | Simple mean pooling |
| `max` | Max pooling |
| `attention` | Single-head additive attention |
| `gated` | Single-head gated attention |
| `abmil` | Attention-based MIL (recommended) |
| `mh_gated4` | Multi-head gated (4 heads) |
| `mh_gated8` | Multi-head gated (8 heads) |
| `temporal_conv32h4` | Temporal conv + gated attention |

### Loss Functions

| Name | Description | Config |
|------|-------------|--------|
| `l1` | MAE (default) | - |
| `l2` / `mse` | MSE | - |
| `huber` | Huber loss | `delta: 2.0` |
| `logcosh` | Log-cosh | - |
| `wing` | Wing loss | `w: 5.0, epsilon: 2.0` |
| `focal` | Focal regression | `gamma: 2.0, base_loss: "l1"` |
| `weighted_zone` | Zone-weighted | `base_loss: "l1"` |
| `threshold_aware` | Clinical thresholds | `thresholds: [5.0], crossing_penalty: 1.0` |
| `combined` | Multiple losses | See examples below |

### Loss Presets

Instead of configuring manually, use presets:

```yaml
loss_preset: "clinical"   # Threshold-aware at clinical cutoffs
# loss_preset: "robust"   # Huber with delta=2.0
# loss_preset: "focal_l1" # Focal with gamma=2.0
```

---

## Examples

### Basic Training

```yaml
# configs/basic.yaml
aggregator: "abmil"
loss_fn: "l1"
use_classifier: true
cls_weight: 0.3
epochs: 100
n_folds: 5
```

```bash
python src/main.py configs/basic.yaml
```

### Huber Loss

```yaml
# configs/huber.yaml
aggregator: "abmil"
loss_fn: "huber"
loss_config:
  delta: 2.0
```

### Focal Loss (Hard Examples)

```yaml
# configs/focal.yaml
aggregator: "abmil"
loss_fn: "focal"
loss_config:
  gamma: 2.0
  base_loss: "l1"
```

### Clinical Loss (Threshold-Aware)

```yaml
# configs/clinical.yaml
aggregator: "abmil"
loss_preset: "clinical"
```

### Combined Loss

```yaml
# configs/combined.yaml
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

### Regression Only (No Classification)

```yaml
# configs/regression_only.yaml
aggregator: "abmil"
use_classifier: false
loss_fn: "l1"
```

### Heavy Classification Weight

```yaml
# configs/cls_heavy.yaml
aggregator: "abmil"
use_classifier: true
cls_weight: 0.7    # 70% classification, 30% regression
```

---

## Running Loss Function Ablations

To compare different loss functions, create multiple config files and run them.

### Method 1: Manual Configs

```bash
# Create configs for each loss
cp configs/default.yaml configs/ablation_l1.yaml
cp configs/default.yaml configs/ablation_huber.yaml
cp configs/default.yaml configs/ablation_focal.yaml

# Edit each config with different loss settings, then run:
python src/main.py configs/ablation_l1.yaml
python src/main.py configs/ablation_huber.yaml
python src/main.py configs/ablation_focal.yaml
```

### Method 2: Shell Script

Create `run_ablation.sh`:

```bash
#!/bin/bash
# Loss function ablation study

LOSSES=("l1" "huber" "focal" "logcosh")

for loss in "${LOSSES[@]}"; do
    echo "========================================"
    echo "Running: $loss"
    echo "========================================"

    # Create temp config
    cat > /tmp/ablation_${loss}.yaml << EOF
video_folder: "/path/to/videos"
csv_path: "/path/to/data.csv"
cache_dir: "/path/to/cache"
output_dir: "/path/to/runs"

aggregator: "abmil"
img_size: 384
n_frames: 175

use_classifier: true
cls_weight: 0.3

loss_fn: "${loss}"
loss_config: {}

epochs: 100
n_folds: 5
seed: 42
EOF

    python src/main.py /tmp/ablation_${loss}.yaml
done

echo "Ablation complete!"
```

Run:
```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

### Method 3: Python Script

Create `run_ablation.py`:

```python
#!/usr/bin/env python3
import subprocess
import yaml

# Base configuration
BASE = {
    "video_folder": "/path/to/videos",
    "csv_path": "/path/to/data.csv",
    "cache_dir": "/path/to/cache",
    "output_dir": "/path/to/runs",
    "aggregator": "abmil",
    "img_size": 384,
    "n_frames": 175,
    "use_classifier": True,
    "cls_weight": 0.3,
    "epochs": 100,
    "n_folds": 5,
    "seed": 42,
}

# Loss configurations to compare
ABLATIONS = {
    "l1": {"loss_fn": "l1"},
    "l2": {"loss_fn": "l2"},
    "huber_1": {"loss_fn": "huber", "loss_config": {"delta": 1.0}},
    "huber_2": {"loss_fn": "huber", "loss_config": {"delta": 2.0}},
    "huber_5": {"loss_fn": "huber", "loss_config": {"delta": 5.0}},
    "focal_1": {"loss_fn": "focal", "loss_config": {"gamma": 1.0, "base_loss": "l1"}},
    "focal_2": {"loss_fn": "focal", "loss_config": {"gamma": 2.0, "base_loss": "l1"}},
    "logcosh": {"loss_fn": "logcosh"},
    "clinical": {"loss_preset": "clinical"},
    "zone_weighted": {"loss_preset": "zone_weighted"},
}

# Run each configuration
for name, settings in ABLATIONS.items():
    print(f"\n{'='*50}")
    print(f"Running: {name}")
    print(f"{'='*50}\n")

    # Merge base config with ablation settings
    cfg = {**BASE, **settings, "run_name": f"ablation_{name}"}

    # Save temporary config
    cfg_path = f"/tmp/ablation_{name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    # Run experiment
    subprocess.run(["python", "src/main.py", cfg_path])

print("\nAblation study complete!")
```

Run:
```bash
python run_ablation.py
```

---

## Classification Head

The model supports multi-task learning with both regression and classification.

### PDFF Stages (4 Classes)

| Class | Stage | PDFF Range |
|-------|-------|------------|
| 0 | Normal | < 6.4% |
| 1 | Mild steatosis | 6.4% - 16.3% |
| 2 | Moderate steatosis | 16.3% - 20.7% |
| 3 | Severe steatosis | > 20.7% |

### Adjusting Loss Balance

```yaml
# More regression focus (default)
cls_weight: 0.3   # 30% cls, 70% reg

# Balanced
cls_weight: 0.5   # 50% cls, 50% reg

# More classification focus
cls_weight: 0.7   # 70% cls, 30% reg

# Regression only
use_classifier: false
```

---

## Output Structure

Each experiment creates:

```
runs/abmil_s384_f175_cls30_CV5/
├── config.json              # Saved configuration
├── train_log.csv            # Training metrics
├── train_log.txt            # Training log
├── train_log.png            # Loss curves
├── best.pt                  # Best model checkpoint
├── fold00/                  # Per-fold results
│   ├── val_predictions.csv
│   ├── test_predictions.csv
│   ├── val_summary.png
│   └── test_summary.png
├── fold01/
├── fold02/
└── ...
```

---

## Cache Generation

Pre-compute frame cache for faster training:

```bash
python src/precompute_cache.py --img_size 384
```

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
decord  # optional, faster video loading
```

---

## Tips

1. **Start with defaults**: Copy `configs/default.yaml` as your template
2. **Use presets**: `loss_preset: "clinical"` is often effective
3. **Ablation order**: Start with `l1`, then try `huber`, `focal`, `clinical`
4. **Classification weight**: 0.3 is a good starting point
5. **Check logs**: Look at `train_log.txt` for training progress
6. **Compare results**: Check `val_predictions.csv` for detailed analysis
