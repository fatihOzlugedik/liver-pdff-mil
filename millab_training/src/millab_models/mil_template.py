"""
Abstract base class for MIL models.
Source: MIL-Lab/src/models/mil_template.py (cleaned for standalone use)
"""
import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod


class MIL(ABC, nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward_attention(self, h: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_features(self, h: torch.Tensor, return_attention: bool = False):
        pass

    @abstractmethod
    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, h: torch.Tensor, loss_fn=None, label=None,
                attn_mask=None, return_attention=False, return_slide_feats=False):
        pass

    @staticmethod
    def ensure_batched(tensor, return_was_unbatched=False):
        was_unbatched = False
        while len(tensor.shape) < 3:
            tensor = tensor.unsqueeze(0)
            was_unbatched = True
        if return_was_unbatched:
            return tensor, was_unbatched
        return tensor

    @staticmethod
    def ensure_unbatched(tensor, return_was_batched=False):
        was_batched = True
        while len(tensor.shape) > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
            was_batched = False
        if return_was_batched:
            return tensor, was_batched
        return tensor

    @staticmethod
    def compute_loss(loss_fn, logits, label):
        if loss_fn is None or logits is None:
            return None
        return loss_fn(logits, label)

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def initialize_classifier(self, num_classes: Optional[int] = None):
        if num_classes is None:
            num_classes = self.num_classes
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        nn.init.kaiming_uniform_(self.classifier.weight, nonlinearity='relu')
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
