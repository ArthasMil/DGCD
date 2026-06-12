# Copyright (c) Open-CD. All rights reserved.
from typing import List

import torch
from torch import Tensor

from opencd.registry import MODELS
from .siamencoder_decoder import SiamEncoderDecoder
from .siamencoder_decoder_twobacks_sam import SiamEncoderDecoderTwoBackSAM

@MODELS.register_module()
class TimeTravellingPixels(SiamEncoderDecoder):

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:

        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        img = torch.cat([img_from, img_to], dim=0)
        img_feat = self.backbone(img)[0]
        feat_from, feat_to = torch.split(img_feat, img_feat.shape[0] // 2, dim=0)
        feat_from = [feat_from]
        feat_to = [feat_to]
        if self.with_neck:
            x = self.neck(feat_from, feat_to)
        else:
            raise ValueError('`NECK` is needed for `TimeTravellingPixels`.')

        return x

@MODELS.register_module()
class TimeTravellingPixels_SAM(SiamEncoderDecoderTwoBackSAM):

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:

        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        img = torch.cat([img_from, img_to], dim=0)
        img_feat = self.backbone(img)[0]
        feat_from, feat_to = torch.split(img_feat, img_feat.shape[0] // 2, dim=0)
        feat_from = [feat_from]
        feat_to = [feat_to]
        if self.with_neck:
            x = self.neck(feat_from, feat_to)
        else:
            raise ValueError('`NECK` is needed for `TimeTravellingPixels`.')

        return x
