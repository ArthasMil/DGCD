
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencd.registry import MODELS
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.losses import CrossEntropyLoss
from ..losses import CrossModalContrastiveLoss

class DepthGuidedMaskRefiner(nn.Module):
    def __init__(self,
                 in_channels=7,
                 depth_channels=64,
                 refine_channels=64,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.in_channels = in_channels
        self.depth_channels = depth_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.logit_branch = nn.Sequential(
            nn.Conv2d(in_channels, refine_channels, 3, padding=1, bias=False if norm_cfg else True),
            nn.BatchNorm2d(refine_channels) if (norm_cfg and norm_cfg['type'] == 'BN') else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.depth_branch = nn.Sequential(
            nn.Conv2d(depth_channels, refine_channels, 3, padding=1, bias=False if norm_cfg else True),
            nn.BatchNorm2d(refine_channels) if (norm_cfg and norm_cfg['type'] == 'BN') else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.fusion_branch = nn.Sequential(
            nn.Conv2d(refine_channels * 2, refine_channels, 3, padding=1, bias=False if norm_cfg else True),
            nn.BatchNorm2d(refine_channels) if (norm_cfg and norm_cfg['type'] == 'BN') else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_channels, in_channels, 1, padding=0, bias=True)
        )

        self.refine_weight = nn.Parameter(torch.tensor(0.3))
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, mask_logit, depth_feat):

        assert mask_logit.dim() == 4 and depth_feat.dim() == 4
        assert mask_logit.device == depth_feat.device
        B, C, H, W = mask_logit.shape

        if depth_feat.shape[2:] != (H, W):
            depth_feat = F.interpolate(
                depth_feat, size=(H, W), mode='bilinear', align_corners=False, recompute_scale_factor=False
            ).contiguous()

        logit_feat = self.logit_branch(mask_logit)
        depth_feat = self.depth_branch(depth_feat)
        fusion_feat = torch.cat([logit_feat, depth_feat], dim=1)
        refine_correction = self.fusion_branch(fusion_feat)

        learnable_weight = self.sigmoid(self.refine_weight)

        refined_logit = mask_logit + learnable_weight * refine_correction

        return refined_logit

@MODELS.register_module()
class ContrastiveUPerHeadWithMaskRefine(UPerHead):
    def __init__(self,
                 **kwargs):

        contrast_losses = kwargs.pop('contrast_losses', [])
        branch_channels = kwargs.pop('branch_channels', [])
        mask_refiner_cfg = kwargs.pop('mask_refiner_cfg', None)
        self.refine_scale = kwargs.pop('refine_scale', 0)
        self.refine_loss_weight = kwargs.pop('refine_loss_weight', 1.0)

        super().__init__(**kwargs)

        self.branch_channels = branch_channels
        self.num_scales = len(branch_channels) if branch_channels else 0
        self.num_classes = kwargs.get('num_classes', 7)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.contrast_losses = nn.ModuleList()
        if self.num_scales > 0 and len(contrast_losses) >= self.num_scales:
            for i in range(self.num_scales):
                c_rgb, c_depth = branch_channels[i]
                loss_cfg = contrast_losses[i]
                loss_cfg['in_channels1'] = c_rgb
                loss_cfg['in_channels2'] = c_depth
                self.contrast_losses.append(MODELS.build(loss_cfg))

        self.mask_refiner = None
        if mask_refiner_cfg is not None and self.num_scales > 0:
            self.refine_scale = min(self.refine_scale, self.num_scales - 1)
            refine_c_rgb, refine_c_depth = branch_channels[self.refine_scale]
            mask_refiner_cfg['in_channels'] = self.num_classes
            mask_refiner_cfg['depth_channels'] = refine_c_depth
            mask_refiner_cfg['refine_channels'] = 64
            self.mask_refiner = DepthGuidedMaskRefiner(**mask_refiner_cfg).to(self.device)

        self.ce_loss = CrossEntropyLoss(
            use_sigmoid=False, reduction='mean', class_weight=kwargs.get('loss_decode', {}).get('class_weight', None)
        ).to(self.device)
        self.ce_loss_weight = kwargs.get('loss_decode', {}).get('loss_weight', 1.0)

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)), f"inputs must be list/tuple, got {type(inputs)}"
        assert len(inputs) == self.num_scales, f"inputs length {len(inputs)} != num_scales {self.num_scales}"

        split_feats_list = []
        refine_depth_feat = None
        for i in range(self.num_scales):
            x = inputs[i]
            assert x.dim() == 4, f"input scale {i} must be 4D, got {x.dim()}D"
            c_rgb, c_depth = self.branch_channels[i]
            assert x.shape[1] == c_rgb + c_depth, f"scale {i} channels {x.shape[1]} != {c_rgb}+{c_depth}"

            feat_rgb = x[:, :c_rgb, :, :].contiguous()
            feat_depth = x[:, c_rgb:, :, :].contiguous()
            split_feats_list.append((feat_rgb, feat_depth))

            if i == self.refine_scale:
                refine_depth_feat = feat_depth

        raw_logit = super().forward(inputs)
        assert raw_logit.dim() == 4, f"raw_logit must be 4D, got {raw_logit.dim()}D"

        refined_logit = raw_logit.clone()
        if self.mask_refiner is not None and refine_depth_feat is not None:
            try:

                refine_depth_feat_up = F.interpolate(
                    refine_depth_feat, size=raw_logit.shape[2:], mode='bilinear', align_corners=False, recompute_scale_factor=False
                ).contiguous()

                refined_logit = self.mask_refiner(
                    raw_logit.to(self.device),
                    refine_depth_feat_up.to(self.device)
                )
            except Exception as e:
                print(f"Refinement error: {e}")

        return refined_logit, raw_logit, split_feats_list

    def predict(self, inputs, batch_img_metas, test_cfg):
        refined_logit, _, _ = self.forward(inputs)
        return self.predict_by_feat(refined_logit, batch_img_metas)

    def loss_by_feat(self, outputs, batch_data_samples, **kwargs):
        refined_logit, raw_logit, split_feats_list = outputs

        labels = self._get_targets(batch_data_samples, **kwargs)
        labels = labels.to(self.device)
        labels = torch.clamp(labels, min=0, max=self.num_classes-1)

        B, C, H, W = refined_logit.shape
        B_label, H_label, W_label = labels.shape
        assert B == B_label, f"batch size mismatch: logit {B} vs label {B_label}"

        if (H, W) != (H_label, W_label):
            refined_logit = F.interpolate(
                refined_logit, size=(H_label, W_label), mode='bilinear', align_corners=False, recompute_scale_factor=False
            ).contiguous()
            raw_logit = F.interpolate(
                raw_logit, size=(H_label, W_label), mode='bilinear', align_corners=False, recompute_scale_factor=False
            ).contiguous()

        loss_raw_ce = torch.tensor(0.0, device=self.device)
        loss_refined_ce = torch.tensor(0.0, device=self.device)
        if raw_logit.numel() > 0 and labels.numel() > 0:
            loss_raw_ce = self.ce_loss(raw_logit, labels) * self.ce_loss_weight
            loss_refined_ce = self.ce_loss(refined_logit, labels) * self.ce_loss_weight * self.refine_loss_weight

        loss_contrast = torch.tensor(0.0, device=self.device)
        valid_count = 0
        for i, (feat_rgb, feat_depth) in enumerate(split_feats_list):
            if i < len(self.contrast_losses) and feat_rgb.numel() > 0 and feat_depth.numel() > 0:
                try:
                    loss_contrast += self.contrast_losses[i](feat_rgb.to(self.device), feat_depth.to(self.device))
                    valid_count += 1
                except Exception as e:
                    print(f"Contrast loss error (scale {i}): {e}")
        if valid_count > 0:
            loss_contrast /= valid_count

        loss_total = loss_raw_ce + loss_refined_ce
        loss = {
            'loss_raw_ce': loss_raw_ce,
            'loss_refined_ce': loss_refined_ce,
            'loss_contrast': loss_contrast,

        }

        return loss

    def _get_targets(self, batch_data_samples, **kwargs):
        try:
            if isinstance(batch_data_samples, list):
                targets = [sample.gt_sem_seg.data for sample in batch_data_samples]
            else:
                targets = batch_data_samples.gt_sem_seg.data

            targets = torch.stack(targets, dim=0)
            if targets.dim() == 4:
                targets = targets.squeeze(1)
            elif targets.dim() != 3:
                B = len(batch_data_samples) if isinstance(batch_data_samples, list) else batch_data_samples.batch_size
                total_pixels = targets.numel() // B
                H = int(torch.sqrt(torch.tensor(total_pixels)).item())
                W = total_pixels // H
                targets = targets.reshape(B, H, W)

            return targets.long()
        except Exception as e:
            print(f"Label extraction error: {e}")
            B = len(batch_data_samples) if isinstance(batch_data_samples, list) else batch_data_samples.batch_size
            return torch.zeros(B, 1024, 1024, dtype=torch.long)
