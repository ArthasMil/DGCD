# Copyright (c) Open-CD. All rights reserved.
import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from opencd.registry import MODELS

class _FSRelation(BaseModule):

    def __init__(
        self,
        scene_embedding_channels: int,
        in_channels_list: List[int],
        out_channels: int,
        norm_cfg: dict,
        act_cfg: dict = dict(type='ReLU'),
    ) -> None:
        super().__init__()

        self.scene_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(scene_embedding_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                )
                for _ in range(len(in_channels_list))
            ]
        )

        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=c,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)))
            self.feature_reencoders.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=c,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)))

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature: Tensor, features: List[Tensor]) -> List[Tensor]:

        content_feats = [
            c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)
        ]
        scene_feats = [op(scene_feature) for op in self.scene_encoder]
        relations = [
            self.normalizer((sf * cf).sum(dim=1, keepdim=True))
            for sf, cf in zip(scene_feats, content_feats)
        ]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats

class _LightWeightDecoder(BaseModule):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_feature_output_strides: List[int] = [4, 8, 16, 32],
        out_feature_output_stride: int = 4,
        norm_cfg: dict = dict(type='SyncBN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU'),
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()
        for in_feat_os in in_feature_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(
                math.log2(int(out_feature_output_stride))
            )
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(
                nn.Sequential(
                    *[nn.Sequential(
                        ConvModule(
                            in_channels=in_channels if idx == 0 else out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg),
                        (nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0
                            else nn.Identity())) for idx in range(num_layers)]))

    def forward(self, features: List[Tensor]) -> Tensor:
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(features[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        return out_feat

@MODELS.register_module()
class FarSegHead(BaseDecodeHead):

    def __init__(self, fsr_channels, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self._fsr = _FSRelation(
            self.in_channels[-1],
            self.in_channels[:-1],
            fsr_channels,
            self.norm_cfg)

        self._decoder = _LightWeightDecoder(fsr_channels, self.channels)
        self.conv_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=3)

    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)
        feats = inputs[:-1]
        scene_embedding = inputs[-1]
        feats = self._fsr(scene_embedding, feats)
        feats = self._decoder(feats)
        return feats

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
