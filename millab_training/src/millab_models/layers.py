"""
Shared attention layers and MLP builder.
Source: MIL-Lab/src/models/layers.py (cleaned for standalone use)
"""
import torch.nn as nn


def create_mlp(
        in_dim=768,
        hid_dims=[512, 512],
        out_dim=512,
        act=nn.ReLU(),
        dropout=0.,
        end_with_fc=True,
        end_with_dropout=False,
        bias=True
    ):
    layers = []
    if len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        if end_with_dropout:
            layers.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*layers)
    return mlp


class GlobalAttention(nn.Module):
    """Attention Network without Gating (2 fc layers)."""
    def __init__(self, L=1024, D=256, dropout=0., num_classes=1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes),
        )

    def forward(self, x):
        return self.module(x)


class GlobalGatedAttention(nn.Module):
    """Attention Network with Sigmoid Gating (3 fc layers)."""
    def __init__(self, L=1024, D=256, dropout=0., num_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A
