# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional

import torch
from torch import Tensor

from opencd.registry import MODELS
from .siamencoder_decoder import SiamEncoderDecoder
from .siamencoder_decoder_twobacks import SiamEncoderDecoderTwoBack

@MODELS.register_module()
class DIEncoderDecoder(SiamEncoderDecoder):

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:

        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        x = self.backbone(img_from, img_to)
        if self.with_neck:
            x = self.neck(x)
        return x

@MODELS.register_module()
class DIEncoderDecoder_twobacks(SiamEncoderDecoderTwoBack):

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:

        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        x = self.backbone(img_from, img_to)
        x = torch.cat(self.backbone(img_from),self.backbone2(img_to),dim=1)
        if self.with_neck:
            x = self.neck(x)
        return x
