# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn

from mmseg.models.backbones import ResNet
from opencd.registry import MODELS

@MODELS.register_module()
class IA_ResNet(ResNet):
    def __init__(self,
                 interaction_cfg=(None, None, None, None),
                 **kwargs):
        super().__init__(**kwargs)
        assert self.num_stages == len(interaction_cfg),\
            'The length of the `interaction_cfg` should be same as the `num_stages`.'

        self.ccs = []
        for ia_cfg in interaction_cfg:
            if ia_cfg is None:
                ia_cfg = dict(type='TwoIdentity')
            self.ccs.append(MODELS.build(ia_cfg))
        self.ccs = nn.ModuleList(self.ccs)

    def forward(self, x1, x2):
        def _stem_forward(x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            return x

        x1 = _stem_forward(x1)
        x2 = _stem_forward(x2)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            x1, x2 = self.ccs[i](x1, x2)
            if i in self.out_indices:
                outs.append(torch.cat([x1, x2], dim=1))
        return tuple(outs)

@MODELS.register_module()
class IA_ResNetV1c(IA_ResNet):

    def __init__(self, **kwargs):
        super(IA_ResNetV1c, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)

@MODELS.register_module()
class IA_ResNetV1d(IA_ResNet):

    def __init__(self, **kwargs):
        super(IA_ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
