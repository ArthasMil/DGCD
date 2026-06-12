import torch
import torch.nn as nn

from opencd.registry import MODELS

from mmseg.models.decode_heads.uper_head import UPerHead
from ..losses import CrossModalContrastiveLoss

@MODELS.register_module()
class ContrastiveUPerHead(UPerHead):
    def __init__(self,
                 contrast_losses,
                 branch_channels,
                 **kwargs):
        super().__init__(**kwargs)

        self.branch_channels = branch_channels
        self.num_scales = len(branch_channels)

        self.contrast_losses = nn.ModuleList()
        for i in range(self.num_scales):
            c_rgb, c_depth = branch_channels[i]
            loss_cfg = contrast_losses[i]

            loss_cfg['in_channels1'] = c_rgb
            loss_cfg['in_channels2'] = c_depth
            self.contrast_losses.append(MODELS.build(loss_cfg))

    def forward(self, inputs):
        split_feats_list = []
        for i, x in enumerate(inputs):
            c_rgb, _ = self.branch_channels[i]

            feat_rgb = x[:, :c_rgb, :, :]
            feat_depth = x[:, c_rgb:, :, :]
            split_feats_list.append((feat_rgb, feat_depth))

        seg_logit = super().forward(inputs)

        return seg_logit, split_feats_list

    def loss_by_feat(self, seg_logit, batch_data_samples, **kwargs):
        seg_logit_tensor, split_feats_list = seg_logit

        loss = super().loss_by_feat(seg_logit_tensor, batch_data_samples,** kwargs)

        total_contrast_loss = 0.0
        for i, (feat_rgb, feat_depth) in enumerate(split_feats_list):

            contrast_loss = self.contrast_losses[i](feat_rgb, feat_depth)
            total_contrast_loss += contrast_loss

        avg_contrast_loss = total_contrast_loss / self.num_scales

        loss['loss_contrast'] = avg_contrast_loss

        return loss
