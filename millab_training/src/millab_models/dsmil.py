"""
DSMIL — Dual-Stream Multiple Instance Learning.
Source: MIL-Lab/src/models/dsmil.py (cleaned for standalone use)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .mil_template import MIL
from .layers import create_mlp


class IClassifier(nn.Module):
    """Instance-level classifier."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.inst_classifier = nn.Linear(in_dim, num_classes)

    def forward(self, h):
        return self.inst_classifier(h)  # B x M x C


class BClassifier(nn.Module):
    """Bag-level classifier with attention."""
    def __init__(self, in_dim, attn_dim=384, dropout=0.0):
        super().__init__()
        self.q = nn.Linear(in_dim, attn_dim)
        self.v = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, in_dim))
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, h, c, attn_mask=None):
        device = h.device
        V = self.v(h)   # B x M x D
        Q = self.q(h)   # B x M x D_attn
        _, m_indices = torch.sort(c, dim=1, descending=True)
        m_feats = torch.stack(
            [torch.index_select(h_i, dim=0, index=m_indices_i[0, :])
             for h_i, m_indices_i in zip(h, m_indices)], 0
        )
        q_max = self.q(m_feats)  # B x C x D_attn
        A = torch.bmm(Q, q_max.transpose(1, 2))  # B x M x C
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=2) * torch.finfo(A.dtype).min
        A = F.softmax(
            A / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32, device=device)),
            dim=1,
        )
        B = torch.bmm(A.transpose(1, 2), V)  # B x C x D
        B = self.norm(B)
        return B, A


class DSMIL(MIL):
    def __init__(self, in_dim=1024, embed_dim=512, num_fc_layers=1,
                 dropout=0.25, attn_dim=384, dropout_v=0.0, num_classes=2):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False,
        )
        self.i_classifier = IClassifier(in_dim=embed_dim, num_classes=num_classes)
        self.b_classifier = BClassifier(in_dim=embed_dim, attn_dim=attn_dim, dropout=dropout_v)
        self.classifier = nn.Conv1d(num_classes, num_classes, kernel_size=embed_dim)
        self.initialize_weights()

    def forward_features(self, h, attn_mask=None, return_attention=False):
        h = self.patch_embed(h)
        instance_classes = self.i_classifier(h)
        slide_feats, attention = self.b_classifier(h, instance_classes, attn_mask=attn_mask)
        intermeds = {'instance_classes': instance_classes}
        if return_attention:
            intermeds['attention'] = attention
        return slide_feats, intermeds

    def forward_attention(self, h, attn_mask=None, attn_only=True):
        pass

    def initialize_classifier(self, num_classes: Optional[int] = None):
        self.classifier = nn.Conv1d(num_classes, num_classes, kernel_size=self.embed_dim)

    def forward_head(self, slide_feats):
        logits = self.classifier(slide_feats)  # B x C x 1
        return logits.squeeze(-1)

    def forward(self, h, label=None, loss_fn=None, attn_mask=None,
                return_attention=False, return_slide_feats=False):
        slide_feats, intermeds = self.forward_features(
            h, attn_mask=attn_mask, return_attention=return_attention
        )
        max_instance_logits, _ = torch.max(intermeds['instance_classes'], 1)
        bag_logits = self.forward_head(slide_feats)
        logits = 0.5 * (bag_logits + max_instance_logits)
        cls_loss = self.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict = {'loss': cls_loss.item() if cls_loss is not None else -1}
        if return_attention:
            log_dict['attention'] = intermeds.get('attention')
        if return_slide_feats:
            log_dict['slide_feats'] = slide_feats
        return results_dict, log_dict
