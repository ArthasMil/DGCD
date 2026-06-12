
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencd.registry import MODELS
from mmseg.models.losses.utils import weighted_loss

@weighted_loss
def contrastive_loss(feat1, feat2, temperature=0.07, num_negatives=128):
    b, c, h, w = feat1.shape
    L = h * w

    feat1 = feat1.flatten(2).transpose(1, 2)
    feat2 = feat2.flatten(2).transpose(1, 2)
    feat1 = F.normalize(feat1, dim=2, p=2)
    feat2 = F.normalize(feat2, dim=2, p=2)

    positive_sim = torch.sum(feat1 * feat2, dim=2)
    positive_sim = (positive_sim + 1.0) / 2.0
    positive_sim = positive_sim / temperature

    neg_indices = torch.empty(b, L, num_negatives, dtype=torch.long, device=feat1.device)

    total_loss = 0.0
    for i in range(b):
        all_sims = torch.matmul(feat1[i], feat2[i].t())
        all_sims = (all_sims + 1.0) / 2.0

        mask = torch.eye(L, device=all_sims.device, dtype=torch.bool)
        min_val = torch.finfo(all_sims.dtype).min
        neg_sims = all_sims.masked_fill(mask, min_val * 0.1)

        neg_sims_scaled = neg_sims / temperature

        neg_indices[i] = torch.multinomial(F.softmax(neg_sims_scaled, dim=1), num_negatives, replacement=True)

    for i in range(b):
        all_sims = torch.matmul(feat1[i], feat2[i].t())
        all_sims = (all_sims + 1.0) / 2.0
        neg_sims = all_sims / temperature

        sampled_neg_sims = torch.gather(neg_sims, 1, neg_indices[i])

        logits = torch.cat([positive_sim[i].unsqueeze(1), sampled_neg_sims], dim=1)
        labels = torch.zeros(L, dtype=torch.long, device=logits.device)
        total_loss += F.cross_entropy(logits, labels)

    return total_loss / b

@MODELS.register_module()
class CrossModalContrastiveLoss(nn.Module):
    def __init__(self,
                 in_channels1,
                 in_channels2,
                 out_channels=64,
                 temperature=0.07,
                 reduction='mean',
                 loss_weight=1.0,
                 downsample_ratio=4,
                 initial_fusion_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.downsample_ratio = downsample_ratio

        initial_logit = torch.log(torch.tensor(initial_fusion_weight) / (1 - torch.tensor(initial_fusion_weight)) + 1e-8)
        self.fusion_weight_logit = nn.Parameter(initial_logit)

        self.proj_head1 = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.proj_head2 = nn.Sequential(
            nn.Conv2d(in_channels2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, feat1, feat2, weight=None, avg_factor=None, reduction_override=None):
        reduction = reduction_override or self.reduction

        if self.downsample_ratio > 1:
            feat1 = F.adaptive_avg_pool2d(feat1, (feat1.shape[2]//self.downsample_ratio, feat1.shape[3]//self.downsample_ratio))
            feat2 = F.adaptive_avg_pool2d(feat2, (feat2.shape[2]//self.downsample_ratio, feat2.shape[3]//self.downsample_ratio))

        rgb_proj = self.proj_head1(feat1)
        depth_proj = self.proj_head2(feat2)

        fusion_weight = torch.sigmoid(self.fusion_weight_logit)

        enhanced_rgb_proj = rgb_proj * (1 - fusion_weight) + depth_proj * fusion_weight

        loss = self.loss_weight * contrastive_loss(
            enhanced_rgb_proj,
            rgb_proj,
            temperature=self.temperature,
            num_negatives=128,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor
        )

        return loss

class DepthGuidedMaskRefinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, 1)
        )
    def forward(self, mask_pred, depth_feat):
        depth_feat = nn.Conv2d(depth_feat.shape[1], 1, 1)(depth_feat)
        combined = torch.cat([mask_pred, depth_feat], dim=1)
        refined_mask = self.conv(combined)
        return refined_mask
