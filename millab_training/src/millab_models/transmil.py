"""
TransMIL — Transformer-based Multiple Instance Learning.
Source: MIL-Lab/src/models/transmil.py (cleaned for standalone use)
"""
import torch
import torch.nn as nn
import numpy as np
from .nystrom_attention import NystromAttention
from .layers import create_mlp
from .mil_template import MIL


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, num_heads=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attention = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attention(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj  = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(MIL):
    def __init__(self, in_dim, embed_dim, num_fc_layers, dropout,
                 num_attention_layers, num_classes, num_heads=8):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )
        self.pos_layer = PPEG(dim=embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransLayer(dim=embed_dim, num_heads=num_heads)
             for _ in range(num_attention_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def forward_attention(self, h):
        pass

    def forward_features(self, h, return_attention=False):
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        h = self.patch_embed(h)
        h, h_square, w_square = self._square_pad(h)
        h = self._add_cls_token(h)
        h, attn = self._apply_trans_layers(h, h_square, w_square, return_attention)
        wsi_feat = self.norm(h)[:, 0]
        return wsi_feat, attn

    def _apply_trans_layers(self, h, h_square, w_square, return_attention=False):
        intermed_dict = {}
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == 0:
                if return_attention:
                    cls_token = h[:, 0]
                    feats = h[:, 1:]
                    intermed_dict['attention'] = torch.matmul(
                        feats, cls_token.unsqueeze(-1)
                    ).squeeze(-1)
                h = self.pos_layer(h, h_square, w_square)
        return h, intermed_dict

    def _square_pad(self, h):
        H = h.shape[1]
        add_length, h_square, w_square = self._get_square_length(H)
        h = torch.cat([h, h[:, :add_length, :]], dim=1)
        return h, h_square, w_square

    def _add_cls_token(self, h):
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        return h

    def _get_square_length(self, H):
        h_square = int(np.ceil(np.sqrt(H)))
        w_square = int(np.ceil(np.sqrt(H)))
        add_length = h_square * w_square - H
        return add_length, h_square, w_square

    def forward_head(self, wsi_feat):
        return self.classifier(wsi_feat)

    def forward(self, h, loss_fn=None, label=None, attn_mask=None,
                return_attention=False, return_slide_feats=False):
        wsi_feats, intermeds = self.forward_features(h, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = self.compute_loss(loss_fn=loss_fn, label=label, logits=logits)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict = {'loss': cls_loss.item() if cls_loss is not None else -1}
        if return_attention:
            log_dict['attention'] = intermeds.get('attention')
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict
