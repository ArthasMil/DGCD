import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmengine.model import Sequential
from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from opencd.registry import MODELS
from mmseg.structures import SegDataSample
import numpy as np

from ..losses import CrossModalContrastiveLoss

class DepthGuidedMaskRefiner(nn.Module):
    def __init__(self,
                 in_channels,
                 depth_channels,
                 refine_channels=64,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.in_channels = in_channels
        self.depth_channels = depth_channels

        self.depth_proj = ConvModule(
            depth_channels,
            in_channels,
            kernel_size=1,
            norm_cfg=None,
            act_cfg=None)

        self.refine_conv = Sequential(
            ConvModule(
                in_channels * 2,
                refine_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                refine_channels,
                refine_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            Conv2d(refine_channels, 1, kernel_size=1))

    def forward(self, mask_feat, depth_feat):
        B, _, H, W = depth_feat.shape
        Q = mask_feat.shape[0] // B

        depth_feat_proj = self.depth_proj(depth_feat)
        depth_feat_proj = depth_feat_proj.unsqueeze(1).repeat(1, Q, 1, 1, 1)
        depth_feat_proj = depth_feat_proj.view(B*Q, self.in_channels, H, W)

        combined_feat = torch.cat([mask_feat, depth_feat_proj], dim=1)
        refined_mask_logit = self.refine_conv(combined_feat)
        return refined_mask_logit

@MODELS.register_module()
class DepthGuidedMask2FormerHead(Mask2FormerHead):
    def __init__(self,
                 in_channels,
                 branch_channels,
                 contrastive_loss_cfg=None,
                 mask_refiner_cfg=None,
                 refine_scale=0,
                 feat_channels=256,
                 num_classes=7,
                 num_queries=100,
                 point_sample_num=12544,
                 **kwargs):
        kwargs['feat_channels'] = feat_channels
        kwargs['in_channels'] = in_channels
        kwargs['num_classes'] = num_classes
        kwargs['num_queries'] = num_queries
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.branch_channels = branch_channels
        self.num_scales = len(branch_channels)
        self.refine_scale = refine_scale
        self.contrastive_losses = nn.ModuleList()
        self.contrastive_loss_cfg = contrastive_loss_cfg
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.point_sample_num = point_sample_num

        assert self.num_scales == 4, f"branch_channels需配置4个尺度，当前为{self.num_scales}个"
        assert len(self.in_channels) == self.num_scales,\
            f"in_channels长度{len(self.in_channels)}与branch_channels尺度数{self.num_scales}不匹配"
        assert 0 <= self.refine_scale < self.num_scales,\
            f"refine_scale需在[0,3]之间，当前为{self.refine_scale}"

        for idx, (c_rgb, c_depth) in enumerate(branch_channels):
            assert c_rgb == self.in_channels[idx],\
                f"尺度{idx}：branch_channels的RGB通道数{c_rgb}与in_channels[{idx}]={self.in_channels[idx]}不匹配"

        if self.contrastive_loss_cfg is not None:
            for (c_rgb, c_depth) in branch_channels:
                self.contrastive_losses.append(
                    CrossModalContrastiveLoss(
                        in_channels1=c_rgb,
                        in_channels2=c_depth,
                        **self.contrastive_loss_cfg
                    )
                )

        self.mask_refiner = None
        if mask_refiner_cfg is not None:
            refine_c_depth = branch_channels[refine_scale][1]
            self.mask_refiner = DepthGuidedMaskRefiner(
                in_channels=self.feat_channels,
                depth_channels=refine_c_depth,
                **mask_refiner_cfg
            )

    def _point_sample_pytorch(self, masks, points, align_corners=False):
        B, C, H, W = masks.shape
        N = points.shape[1]

        grid = points.clone()
        grid[:, :, 0] = grid[:, :, 0] * 2 - 1
        grid[:, :, 1] = grid[:, :, 1] * 2 - 1
        grid = grid.unsqueeze(1)

        sampled = F.grid_sample(
            masks,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=align_corners
        )

        return sampled.squeeze(2)

    def _get_targets_single(self, img_meta, gt_semantic_seg, point_coords, num_queries):
        H, W = gt_semantic_seg.shape[:2]
        device = gt_semantic_seg.device

        classes = torch.unique(gt_semantic_seg)
        classes = classes[classes != 0]
        if len(classes) == 0:

            gt_labels = torch.zeros(num_queries, dtype=torch.long, device=device)
            gt_masks = torch.zeros(num_queries, H, W, dtype=torch.bool, device=device)
            gt_areas = torch.zeros(num_queries, dtype=torch.float32, device=device)
            return gt_labels, gt_masks, gt_areas

        class_masks = []
        class_areas = []
        for cls in classes:
            mask = (gt_semantic_seg == cls).bool()
            area = mask.sum().float()
            if area > 0:
                class_masks.append(mask)
                class_areas.append(area)
        class_masks = torch.stack(class_masks, dim=0)
        class_areas = torch.tensor(class_areas, device=device)

        point_coords = point_coords.clone().unsqueeze(0)
        class_masks_input = class_masks.unsqueeze(1).float()

        point_masks = self._point_sample_pytorch(
            class_masks_input,
            point_coords,
            align_corners=False
        )
        point_masks = point_masks.squeeze(1)
        point_masks = point_masks > 0.5

        sorted_indices = torch.argsort(class_areas, descending=True)
        gt_labels = torch.zeros(num_queries, dtype=torch.long, device=device)
        gt_masks = torch.zeros(num_queries, H, W, dtype=torch.bool, device=device)
        gt_areas = torch.zeros(num_queries, dtype=torch.float32, device=device)

        for i in range(min(num_queries, len(sorted_indices))):
            cls_idx = sorted_indices[i]
            gt_labels[i] = classes[cls_idx]
            gt_masks[i] = class_masks[cls_idx]
            gt_areas[i] = class_areas[cls_idx]

        if num_queries > len(classes):
            gt_labels[len(classes):] = 0
            gt_masks[len(classes):] = False
            gt_areas[len(classes):] = 0

        return gt_labels, gt_masks, gt_areas

    def forward(self, inputs, batch_data_samples):
        B = inputs[0].shape[0]
        rgb_feats = []
        depth_feats = []
        shallow_depth_feat = None

        for scale_idx, (feat, (c_rgb, c_depth)) in enumerate(zip(inputs, self.branch_channels)):
            assert feat.shape[1] == c_rgb + c_depth,\
                f"尺度{scale_idx}：输入通道数{feat.shape[1]} != RGB({c_rgb})+Depth({c_depth})"

            rgb_feat = feat[:, :c_rgb, :, :]
            depth_feat = feat[:, c_rgb:c_rgb+c_depth, :, :]

            rgb_feats.append(rgb_feat)
            depth_feats.append(depth_feat)

            if scale_idx == self.refine_scale:
                shallow_depth_feat = depth_feat

        all_cls_scores, all_mask_preds = super().forward(rgb_feats, batch_data_samples)

        contrast_loss = None
        if len(self.contrastive_losses) > 0:
            scale_losses = []
            for idx in range(self.num_scales):
                cl = self.contrastive_losses[idx](rgb_feats[idx], depth_feats[idx])
                scale_losses.append(cl)
            contrast_loss = sum(scale_losses) / self.num_scales

        if self.mask_refiner is not None and shallow_depth_feat is not None:
            final_mask_preds = all_mask_preds[-1]
            B, Q, H, W = final_mask_preds.shape

            mask_feat = final_mask_preds.permute(0, 2, 3, 1).reshape(B, H*W, Q)
            mask_feat = mask_feat.permute(0, 2, 1).reshape(B*Q, 1, H, W)

            if mask_feat.shape[1] != self.feat_channels:
                mask_feat = Conv2d(1, self.feat_channels, kernel_size=1).to(mask_feat.device)(mask_feat)

            shallow_depth_feat = F.interpolate(
                shallow_depth_feat, size=(H, W), mode='bilinear', align_corners=False
            )

            refined_mask_logits = self.mask_refiner(mask_feat, shallow_depth_feat)
            refined_mask_logits = refined_mask_logits.view(B, Q, H, W)

            all_mask_preds = list(all_mask_preds)
            all_mask_preds[-1] = refined_mask_logits

        return all_cls_scores, all_mask_preds, contrast_loss

    def loss(self, inputs, batch_data_samples, train_cfg):
        all_cls_scores, all_mask_preds, contrast_loss = self(inputs, batch_data_samples)

        if isinstance(batch_data_samples, list):
            batch_img_metas = [
                sample.metainfo.get('img_meta', {}) for sample in batch_data_samples
            ]
        else:
            batch_img_metas = batch_data_samples.get('metainfo', {}).get('img_meta', {})
            if not isinstance(batch_img_metas, list):
                batch_img_metas = [batch_img_metas]

        losses = super().loss_by_feat(
            all_cls_scores,
            all_mask_preds,
            batch_data_samples,
            batch_img_metas
        )

        if contrast_loss is not None and hasattr(self.contrastive_losses[0], 'loss_weight'):
            losses['contrast_loss'] = contrast_loss * self.contrastive_losses[0].loss_weight

        return losses

    def predict(self, inputs, batch_img_metas, test_cfg):
        batch_data_samples = [
            SegDataSample(metainfo={'img_meta': meta}) for meta in batch_img_metas
        ]
        all_cls_scores, all_mask_preds, _ = super().forward(inputs, batch_data_samples)
        return super().predict_by_feat(all_cls_scores, all_mask_preds, batch_img_metas, test_cfg)
