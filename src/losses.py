"""
Loss functions for PDFF (fat percentage) regression from USG videos with MIL.

This module provides multiple configurable loss functions suitable for:
- Regression tasks with bounded targets (0-100% PDFF)
- Multiple Instance Learning (MIL) scenarios
- Handling outliers and class imbalance in PDFF distribution

Usage:
    from losses import build_loss
    loss_fn = build_loss("huber", delta=1.0)
    loss = loss_fn(predictions, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


# =============================================================================
# BASE REGRESSION LOSSES
# =============================================================================

class L1Loss(nn.Module):
    """Mean Absolute Error (MAE) - robust to outliers."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class L2Loss(nn.Module):
    """Mean Squared Error (MSE) - penalizes large errors more."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class RMSELoss(nn.Module):
    """Root Mean Squared Error - same scale as target."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(pred, target) + self.eps)


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1) - combines L1 and L2.
    L2 for |error| < delta, L1 otherwise.

    Args:
        delta: Threshold for switching between L1 and L2.
               For PDFF: delta=1.0-2.0 is typical (1-2% error threshold).
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


# =============================================================================
# ADVANCED REGRESSION LOSSES
# =============================================================================

class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss: log(cosh(error))

    Smooth approximation that behaves like L2 for small errors
    and L1 for large errors. Twice differentiable everywhere.
    Good for PDFF regression where we want smooth gradients.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        return torch.mean(torch.log(torch.cosh(error + 1e-12)))


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for asymmetric error penalization.

    Args:
        quantile: Target quantile (0.5 = median = MAE).
                  quantile > 0.5: penalize underestimation more
                  quantile < 0.5: penalize overestimation more

    For PDFF: Use quantile > 0.5 if missing high-fat cases is more costly.
    """
    def __init__(self, quantile: float = 0.5):
        super().__init__()
        assert 0 < quantile < 1, "Quantile must be in (0, 1)"
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - pred
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        return torch.mean(loss)


class WingLoss(nn.Module):
    """
    Wing Loss from facial landmark detection.

    Logarithmic for small errors (|x| < w), linear for large errors.
    Better gradient behavior for small errors than L1.

    Args:
        w: Width of non-linear region (small errors)
        epsilon: Curvature of non-linear region

    For PDFF: w=5.0 (5% threshold), epsilon=2.0 works well.
    """
    def __init__(self, w: float = 5.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * math.log(1 + self.w / self.epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.abs(pred - target)

        # Small errors: logarithmic
        small_mask = x < self.w
        loss_small = self.w * torch.log(1 + x / self.epsilon)

        # Large errors: linear
        loss_large = x - self.C

        loss = torch.where(small_mask, loss_small, loss_large)
        return torch.mean(loss)


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss - improved version of Wing Loss.

    Adapts curvature based on error magnitude.
    Better for regression with varying error scales.

    Args:
        omega: Controls curvature (higher = more curved)
        theta: Threshold for linear region
        epsilon: Smoothing constant
        alpha: Power for adaptive curvature
    """
    def __init__(self, omega: float = 14.0, theta: float = 0.5,
                 epsilon: float = 1.0, alpha: float = 2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(pred - target)

        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - delta))) * \
            (self.alpha - delta) * ((self.theta / self.epsilon) ** (self.alpha - delta - 1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - delta))

        small_mask = delta < self.theta
        loss_small = self.omega * torch.log(1 + (delta / self.epsilon) ** (self.alpha - delta))
        loss_large = A * delta - C

        loss = torch.where(small_mask, loss_small, loss_large)
        return torch.mean(loss)


# =============================================================================
# PERCENTAGE/BOUNDED REGRESSION LOSSES
# =============================================================================

class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error.

    Useful when relative error matters more than absolute error.

    Args:
        epsilon: Small constant to avoid division by zero for targets near 0.

    Note: For PDFF, targets can be 0% (no fat), so epsilon is important.
    """
    def __init__(self, epsilon: float = 1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((target - pred) / (torch.abs(target) + self.epsilon)))


class SMAPELoss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error.

    Symmetric version of MAPE that handles zero targets better.
    Range: [0, 2] (or [0, 200%])
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = torch.abs(pred - target)
        denominator = (torch.abs(pred) + torch.abs(target)) / 2 + self.epsilon
        return torch.mean(numerator / denominator)


class BoundedRegressionLoss(nn.Module):
    """
    Loss that respects bounded output range [0, max_value].

    Combines base loss with penalty for out-of-bounds predictions.

    Args:
        base_loss: Underlying loss function
        min_val: Minimum valid value (0 for PDFF)
        max_val: Maximum valid value (100 for PDFF percentage)
        penalty_weight: Weight for boundary penalty
    """
    def __init__(self, base_loss: str = "l1", min_val: float = 0.0,
                 max_val: float = 100.0, penalty_weight: float = 0.1):
        super().__init__()
        self.base_loss = build_loss(base_loss)
        self.min_val = min_val
        self.max_val = max_val
        self.penalty_weight = penalty_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(pred, target)

        # Penalty for predictions outside bounds
        below_min = F.relu(self.min_val - pred)
        above_max = F.relu(pred - self.max_val)
        penalty = torch.mean(below_min + above_max)

        return base + self.penalty_weight * penalty


# =============================================================================
# FOCAL/WEIGHTED REGRESSION LOSSES
# =============================================================================

class FocalRegressionLoss(nn.Module):
    """
    Focal Loss adapted for regression.

    Down-weights easy examples (small errors), focuses on hard examples.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        base_loss: Base loss type ("l1" or "l2")
        normalize: Whether to normalize by sum of weights

    For PDFF: gamma=2.0 helps focus on difficult cases (intermediate fat %).
    """
    def __init__(self, gamma: float = 2.0, base_loss: str = "l1",
                 normalize: bool = True):
        super().__init__()
        self.gamma = gamma
        self.base_loss = base_loss
        self.normalize = normalize

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.base_loss == "l1":
            error = torch.abs(pred - target)
        else:
            error = (pred - target) ** 2

        # Modulating factor: (1 - exp(-error))^gamma
        # Large errors -> weight ~ 1, Small errors -> weight ~ 0
        weight = (1 - torch.exp(-error)) ** self.gamma

        weighted_loss = weight * error

        if self.normalize:
            return weighted_loss.sum() / (weight.sum() + 1e-8)
        return weighted_loss.mean()


class WeightedZoneLoss(nn.Module):
    """
    Loss with different weights for different PDFF zones.

    Useful when certain PDFF ranges are clinically more important
    (e.g., around the 5% threshold for steatosis diagnosis).

    Args:
        base_loss: Underlying loss function
        zones: List of (min, max, weight) tuples defining zones
               Default zones based on clinical PDFF thresholds:
               - Normal: 0-5% (weight 1.0)
               - Mild steatosis: 5-16% (weight 1.5)
               - Moderate: 16-21% (weight 1.5)
               - Severe: >21% (weight 2.0)
    """
    def __init__(self, base_loss: str = "l1",
                 zones: Optional[List[tuple]] = None):
        super().__init__()
        self.base_loss_fn = build_loss(base_loss)

        # Default clinical PDFF zones
        self.zones = zones or [
            (0, 5, 1.0),      # Normal
            (5, 6.4, 1.5),    # Borderline - most critical threshold
            (6.4, 16.3, 1.2), # Mild steatosis
            (16.3, 20.7, 1.3),# Moderate
            (20.7, 100, 1.5), # Severe
        ]

    def _get_weight(self, target: torch.Tensor) -> torch.Tensor:
        weight = torch.ones_like(target)
        for min_val, max_val, w in self.zones:
            mask = (target >= min_val) & (target < max_val)
            weight[mask] = w
        return weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self._get_weight(target)

        # Per-sample loss
        if isinstance(self.base_loss_fn, L1Loss):
            error = torch.abs(pred - target)
        elif isinstance(self.base_loss_fn, L2Loss):
            error = (pred - target) ** 2
        else:
            # For other losses, compute element-wise
            error = torch.abs(pred - target)  # Approximate

        weighted_error = weight * error
        return weighted_error.mean()


class ThresholdAwareLoss(nn.Module):
    """
    Loss that penalizes errors crossing clinical thresholds more.

    For PDFF diagnosis, crossing the 5% threshold (or other thresholds)
    changes the clinical interpretation.

    Args:
        base_loss: Underlying loss function
        thresholds: List of clinical thresholds (default: [5.0, 6.4, 16.3, 20.7])
        crossing_penalty: Additional penalty when pred and target are on different sides
    """
    def __init__(self, base_loss: str = "l1",
                 thresholds: Optional[List[float]] = None,
                 crossing_penalty: float = 1.0):
        super().__init__()
        self.base_loss_fn = build_loss(base_loss)
        self.thresholds = thresholds or [5.0, 6.4, 16.3, 20.7]
        self.crossing_penalty = crossing_penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss_fn(pred, target)

        # Count threshold crossings
        penalty = torch.zeros_like(pred)
        for thresh in self.thresholds:
            # 1 if prediction and target are on different sides of threshold
            crossing = ((pred >= thresh) != (target >= thresh)).float()
            penalty = penalty + crossing

        return base + self.crossing_penalty * penalty.mean()


# =============================================================================
# MIL-SPECIFIC LOSSES
# =============================================================================

class MILRankingLoss(nn.Module):
    """
    Ranking loss for MIL - ensures correct ordering of bag predictions.

    Given bags with different PDFF values, predictions should preserve
    the relative ordering.

    Args:
        margin: Minimum margin between ordered pairs
        base_loss: Additional regression loss (optional)
        ranking_weight: Weight for ranking component

    Note: Requires batch_size > 1 or accumulation of samples.
    """
    def __init__(self, margin: float = 1.0, base_loss: str = "l1",
                 ranking_weight: float = 0.5):
        super().__init__()
        self.margin = margin
        self.base_loss_fn = build_loss(base_loss)
        self.ranking_weight = ranking_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss_fn(pred, target)

        # Ranking loss (requires multiple samples)
        if pred.numel() <= 1:
            return base

        # All pairwise comparisons
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)  # (N, N)
        target_diff = target.unsqueeze(0) - target.unsqueeze(1)  # (N, N)

        # Sign of target difference indicates correct ordering
        sign = torch.sign(target_diff)

        # Margin ranking loss: max(0, -sign * pred_diff + margin)
        # Only for pairs with different targets
        mask = (target_diff.abs() > 1e-6).float()
        ranking = F.relu(-sign * pred_diff + self.margin) * mask
        ranking_loss = ranking.sum() / (mask.sum() + 1e-8)

        return base + self.ranking_weight * ranking_loss


class ContrastiveMILLoss(nn.Module):
    """
    Contrastive loss for MIL feature learning.

    Pulls together bags with similar PDFF, pushes apart dissimilar bags.

    Args:
        margin: Margin for dissimilar pairs
        similarity_threshold: PDFF difference below which bags are "similar"
        base_loss: Regression loss
        contrastive_weight: Weight for contrastive component

    Note: This loss works on features, not predictions. Use with model hooks.
    """
    def __init__(self, margin: float = 1.0, similarity_threshold: float = 3.0,
                 base_loss: str = "l1", contrastive_weight: float = 0.1):
        super().__init__()
        self.margin = margin
        self.similarity_threshold = similarity_threshold
        self.base_loss_fn = build_loss(base_loss)
        self.contrastive_weight = contrastive_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.base_loss_fn(pred, target)

        if features is None or features.size(0) <= 1:
            return base

        # Compute pairwise feature distances
        feat_dist = torch.cdist(features, features, p=2)  # (N, N)

        # Similar pairs (PDFF difference < threshold)
        target_dist = torch.abs(target.unsqueeze(0) - target.unsqueeze(1))
        similar = (target_dist < self.similarity_threshold).float()
        dissimilar = 1 - similar

        # Contrastive loss
        # Similar: minimize distance
        # Dissimilar: maximize distance (up to margin)
        loss_similar = similar * feat_dist ** 2
        loss_dissimilar = dissimilar * F.relu(self.margin - feat_dist) ** 2

        # Exclude diagonal
        n = features.size(0)
        mask = 1 - torch.eye(n, device=features.device)
        contrastive = ((loss_similar + loss_dissimilar) * mask).sum() / (mask.sum() + 1e-8)

        return base + self.contrastive_weight * contrastive


# =============================================================================
# UNCERTAINTY-AWARE LOSSES
# =============================================================================

class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood for uncertainty estimation.

    Model outputs both mean (prediction) and variance (uncertainty).

    Args:
        eps: Minimum variance to prevent log(0)
        reduction: 'mean' or 'sum'

    Note: Requires model to output (mean, log_var) tuple.
    """
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                log_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        if log_var is None:
            # Fall back to L2 loss if no uncertainty provided
            return F.mse_loss(pred, target)

        var = torch.exp(log_var) + self.eps
        loss = 0.5 * (torch.log(var) + (target - pred) ** 2 / var)

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class LaplacianNLLLoss(nn.Module):
    """
    Laplacian Negative Log-Likelihood - MAE with uncertainty.

    Uses Laplace distribution (heavier tails than Gaussian).
    Model outputs mean and log_scale.

    Args:
        eps: Minimum scale
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                log_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if log_scale is None:
            return F.l1_loss(pred, target)

        scale = torch.exp(log_scale) + self.eps
        loss = torch.log(2 * scale) + torch.abs(target - pred) / scale
        return loss.mean()


# =============================================================================
# COMBINED LOSSES
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with configurable weights.

    Args:
        losses: Dict mapping loss names to weights
                e.g., {"l1": 0.5, "huber": 0.3, "ranking": 0.2}
        loss_configs: Optional dict of loss-specific configurations

    Example:
        loss = CombinedLoss(
            losses={"l1": 0.7, "huber": 0.3},
            loss_configs={"huber": {"delta": 2.0}}
        )
    """
    def __init__(self, losses: Dict[str, float],
                 loss_configs: Optional[Dict[str, Dict]] = None):
        super().__init__()
        self.weights = losses
        loss_configs = loss_configs or {}

        self.loss_fns = nn.ModuleDict()
        for name in losses.keys():
            config = loss_configs.get(name, {})
            self.loss_fns[name] = build_loss(name, **config)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        total = 0.0
        for name, weight in self.weights.items():
            loss = self.loss_fns[name](pred, target, **kwargs)
            total = total + weight * loss
        return total


class DynamicWeightedLoss(nn.Module):
    """
    Dynamically weights multiple losses based on their magnitudes.

    Uses uncertainty weighting (Kendall et al., 2018) to automatically
    balance multiple loss terms.

    Args:
        losses: List of loss names to combine
        loss_configs: Optional configurations per loss
    """
    def __init__(self, losses: List[str],
                 loss_configs: Optional[Dict[str, Dict]] = None):
        super().__init__()
        loss_configs = loss_configs or {}

        self.loss_fns = nn.ModuleList([
            build_loss(name, **loss_configs.get(name, {}))
            for name in losses
        ])

        # Learnable log-variances for each loss
        self.log_vars = nn.Parameter(torch.zeros(len(losses)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        total = 0.0
        for i, loss_fn in enumerate(self.loss_fns):
            loss = loss_fn(pred, target, **kwargs)
            # Uncertainty weighting: L / (2 * var) + log(var) / 2
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + 0.5 * self.log_vars[i]
        return total


# =============================================================================
# LOSS FACTORY
# =============================================================================

LOSS_REGISTRY = {
    # Basic
    "l1": L1Loss,
    "mae": L1Loss,
    "l2": L2Loss,
    "mse": L2Loss,
    "rmse": RMSELoss,
    "huber": HuberLoss,
    "smooth_l1": HuberLoss,

    # Advanced
    "logcosh": LogCoshLoss,
    "quantile": QuantileLoss,
    "wing": WingLoss,
    "adaptive_wing": AdaptiveWingLoss,

    # Percentage/Bounded
    "mape": MAPELoss,
    "smape": SMAPELoss,
    "bounded": BoundedRegressionLoss,

    # Focal/Weighted
    "focal": FocalRegressionLoss,
    "weighted_zone": WeightedZoneLoss,
    "threshold_aware": ThresholdAwareLoss,

    # MIL-specific
    "ranking": MILRankingLoss,
    "contrastive": ContrastiveMILLoss,

    # Uncertainty
    "gaussian_nll": GaussianNLLLoss,
    "laplacian_nll": LaplacianNLLLoss,

    # Combined
    "combined": CombinedLoss,
    "dynamic": DynamicWeightedLoss,
}


def build_loss(name: str, **kwargs) -> nn.Module:
    """
    Build a loss function by name.

    Args:
        name: Loss function name (see LOSS_REGISTRY)
        **kwargs: Loss-specific configuration

    Returns:
        Instantiated loss module

    Examples:
        >>> loss = build_loss("l1")
        >>> loss = build_loss("huber", delta=2.0)
        >>> loss = build_loss("combined", losses={"l1": 0.7, "huber": 0.3})
        >>> loss = build_loss("focal", gamma=2.0, base_loss="l1")
    """
    name_lower = name.lower()

    if name_lower not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY.keys()))
        raise ValueError(f"Unknown loss '{name}'. Available: {available}")

    return LOSS_REGISTRY[name_lower](**kwargs)


def get_available_losses() -> List[str]:
    """Return list of available loss function names."""
    return sorted(LOSS_REGISTRY.keys())


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

LOSS_PRESETS = {
    "default": {
        "name": "l1",
        "config": {}
    },
    "robust": {
        "name": "huber",
        "config": {"delta": 2.0}
    },
    "clinical": {
        "name": "threshold_aware",
        "config": {
            "base_loss": "l1",
            "thresholds": [5.0, 6.4, 16.3, 20.7],
            "crossing_penalty": 1.0
        }
    },
    "focal_l1": {
        "name": "focal",
        "config": {"gamma": 2.0, "base_loss": "l1"}
    },
    "combined_robust": {
        "name": "combined",
        "config": {
            "losses": {"l1": 0.5, "huber": 0.3, "logcosh": 0.2},
            "loss_configs": {"huber": {"delta": 2.0}}
        }
    },
    "zone_weighted": {
        "name": "weighted_zone",
        "config": {
            "base_loss": "l1",
            "zones": [
                (0, 5, 1.0),
                (5, 6.4, 2.0),  # Critical threshold
                (6.4, 16.3, 1.2),
                (16.3, 20.7, 1.3),
                (20.7, 100, 1.5),
            ]
        }
    },
    "ranking_l1": {
        "name": "ranking",
        "config": {
            "margin": 1.0,
            "base_loss": "l1",
            "ranking_weight": 0.3
        }
    },
}


def build_loss_from_preset(preset_name: str) -> nn.Module:
    """
    Build a loss function from a preset configuration.

    Args:
        preset_name: Name of the preset (see LOSS_PRESETS)

    Returns:
        Instantiated loss module
    """
    if preset_name not in LOSS_PRESETS:
        available = ", ".join(sorted(LOSS_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    preset = LOSS_PRESETS[preset_name]
    return build_loss(preset["name"], **preset["config"])


def get_available_presets() -> List[str]:
    """Return list of available preset names."""
    return sorted(LOSS_PRESETS.keys())
