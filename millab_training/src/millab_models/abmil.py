"""
ABMIL — Attention-Based Multiple Instance Learning.
Source: MIL-Lab/src/models/abmil.py (cleaned for standalone use)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mil_template import MIL
from .layers import GlobalAttention, GlobalGatedAttention, create_mlp


class ABMIL(MIL):
    def __init__(self, in_dim=1024, embed_dim=512, num_fc_layers=1,
                 dropout=0.25, attn_dim=384, gate=True, num_classes=2):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )
        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def forward_attention(self, h, attn_mask=None, attn_only=True):
        h = self.patch_embed(h)
        A = self.global_attn(h)            # B x M x K
        A = torch.transpose(A, -2, -1)    # B x K x M
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        if attn_only:
            return A
        return h, A

    def forward_features(self, h, attn_mask=None, return_attention=True):
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        A = F.softmax(A_base, dim=-1)
        h = torch.bmm(A, h).squeeze(dim=1)  # B x K x C -> B x C
        log_dict = {'attention': A_base if return_attention else None}
        return h, log_dict

    def forward_head(self, h):
        return self.classifier(h)

    def forward(self, h, loss_fn=None, label=None, attn_mask=None,
                return_attention=False, return_slide_feats=False):
        wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict
