"""
CLAM-SB — Clustering-constrained Attention MIL (Single Branch).
Source: MIL-Lab/src/models/clam.py (cleaned for standalone use)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mil_template import MIL
from .layers import GlobalGatedAttention, GlobalAttention, create_mlp


class CLAMSB(MIL):
    def __init__(self, in_dim=1024, embed_dim=512, n_fc_layers=1,
                 dropout=0.25, gate=True, attention_dim=384,
                 num_classes=2, k_sample=8, subtyping=False,
                 instance_loss_fn='ce', bag_weight=0.7):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.bag_weight = bag_weight

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (n_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )
        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(L=embed_dim, D=attention_dim,
                                      dropout=dropout, num_classes=1)
        self.classifier = nn.Linear(embed_dim, num_classes)

        instance_classifiers = [nn.Linear(embed_dim, 2) for _ in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.initialize_weights()

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, h.device)
        n_targets = self.create_negative_targets(self.k_sample, h.device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, h.device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward_attention(self, h, attention_only=False):
        h = self.patch_embed(h.squeeze(0))
        A = self.global_attn(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        return h, A

    def forward_head(self, h):
        return self.classifier(h)

    def forward_features(self, h, return_attention=True):
        h_embedded, attention = self.forward_attention(h)
        log_dict = {'instance_feats': h_embedded}
        if return_attention:
            log_dict['attention'] = attention
        attention_scaled = F.softmax(attention, dim=-1)
        M = torch.mm(attention_scaled, h_embedded)
        return M, log_dict

    def forward_instance_heads(self, h, attention_scores, label=None):
        if label is None:
            return None
        total_inst_loss = 0.0
        inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze(0).to(label.device)
        for i in range(len(self.instance_classifiers)):
            inst_label_for_class = inst_labels[i].item()
            classifier = self.instance_classifiers[i]
            if inst_label_for_class == 1:
                instance_loss, _, _ = self.inst_eval(attention_scores, h, classifier)
                total_inst_loss += instance_loss
            else:
                if self.subtyping:
                    instance_loss, _, _ = self.inst_eval_out(attention_scores, h, classifier)
                    total_inst_loss += instance_loss
        if self.subtyping and len(self.instance_classifiers) > 0:
            total_inst_loss /= len(self.instance_classifiers)
        elif not self.subtyping and inst_labels.sum().item() > 0:
            total_inst_loss /= inst_labels.sum().item()
        elif total_inst_loss == 0 and inst_labels.sum().item() == 0:
            return None
        return total_inst_loss

    def compute_total_loss(self, logits, label, loss_fn, inst_loss):
        cls_loss = self.compute_loss(loss_fn, logits, label)
        if inst_loss is not None:
            loss = cls_loss * self.bag_weight + (1 - self.bag_weight) * inst_loss
        else:
            loss = cls_loss
        return loss

    def forward(self, h, label=None, loss_fn=None, attn_mask=None,
                return_attention=True, return_slide_feats=None):
        slide_feats, intermeds = self.forward_features(h, return_attention=return_attention)
        logits = self.forward_head(slide_feats)

        # Instance loss only when label and loss_fn are provided (training classification)
        inst_loss = None
        if label is not None and loss_fn is not None:
            inst_loss = self.forward_instance_heads(
                intermeds['instance_feats'], intermeds['attention'], label
            )
            total_loss = self.compute_total_loss(logits, label, loss_fn, inst_loss)
        elif loss_fn is not None and label is not None:
            total_loss = self.compute_loss(loss_fn, logits, label)
        else:
            total_loss = None

        log_dict = {
            'instance_loss': inst_loss.item() if inst_loss is not None else -1,
            'loss': total_loss.item() if total_loss is not None else -1,
        }
        results_dict = {'logits': logits, 'loss': total_loss}
        if return_attention:
            log_dict['attention'] = intermeds.get('attention')
        if return_slide_feats:
            log_dict['slide_feats'] = slide_feats
        return results_dict, log_dict
