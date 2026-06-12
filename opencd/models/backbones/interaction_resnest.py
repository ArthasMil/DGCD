# Copyright (c) Open-CD. All rights reserved.
from mmseg.models.backbones.resnest import Bottleneck
from mmseg.models.utils import ResLayer
from opencd.registry import MODELS
from .interaction_resnet import IA_ResNetV1d

@MODELS.register_module()
class IA_ResNeSt(IA_ResNetV1d):

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }

    def __init__(self,
                 groups=1,
                 base_width=4,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True,
                 **kwargs):
        self.groups = groups
        self.base_width = base_width
        self.radix = radix
        self.reduction_factor = reduction_factor
        self.avg_down_stride = avg_down_stride
        super(IA_ResNeSt, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            radix=self.radix,
            reduction_factor=self.reduction_factor,
            avg_down_stride=self.avg_down_stride,
            **kwargs)
