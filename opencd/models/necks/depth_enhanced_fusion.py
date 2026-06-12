# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from typing import List

from opencd.registry import MODELS

@MODELS.register_module()
class DepthEhancedFeatureFusionNeck(BaseModule):

    def __init__(self,
                 policy,
                 in_channels_rgb: List[int],
                 in_channels_depth: List[int],
                 channels=None,
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.policy = policy
        self.in_channels_rgb = in_channels_rgb
        self.in_channels_depth = in_channels_depth
        self.channels = channels
        self.out_indices = out_indices
        self.num_ins = len(in_channels_rgb)

        self.down_channel = nn.ModuleList()
        self.soft_ffn = nn.ModuleList()

        for i in range(self.num_ins):
            self.down_channel.append(nn.Conv2d(
                self.in_channels_rgb[i] + self.in_channels_depth[i],
                1,
                kernel_size=1,
                stride=1,
                bias=False
            ))
            self.soft_ffn.append(nn.Sequential(
                nn.Conv2d(self.in_channels_depth[i], self.in_channels_depth[i], kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(self.in_channels_depth[i], self.in_channels_rgb[i], kernel_size=1, stride=1),
            ))

    def forward(self, x1, x2):
        assert len(x1) == len(x2), "The features x1 and x2 from the backbone should be of equal length"

        outs = []
        target_device = x1[0].device
        target_dtype = x1[0].dtype

        for i in range(self.num_ins):
            down_channel_layer = self.down_channel[i].to(target_device, target_dtype)
            soft_ffn_module = self.soft_ffn[i].to(target_device, target_dtype)

            rgb_feat = x1[i]
            depth_feat = x2[i]

            x0_1 = torch.cat([rgb_feat, depth_feat], dim=1)

            activate_map = down_channel_layer(x0_1)

            activate_map = torch.sigmoid(activate_map)

            rgb_feat_enhanced = soft_ffn_module(depth_feat * activate_map)

            out = rgb_feat + rgb_feat_enhanced
            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
