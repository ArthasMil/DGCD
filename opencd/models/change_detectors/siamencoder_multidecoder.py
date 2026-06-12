# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.models.utils import resize
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from opencd.registry import MODELS
from .siamencoder_decoder import SiamEncoderDecoder

@MODELS.register_module()
class SiamEncoderMultiDecoder(SiamEncoderDecoder):

    def __init__(self, postprocess_pred_and_label=None, **kwargs):
        super().__init__(**kwargs)
        self.postprocess_pred_and_label = postprocess_pred_and_label

    def _init_decode_head(self, decode_head: ConfigType) -> None:

        self.decode_head = MODELS.build(decode_head)
        self.num_classes = self.decode_head.binary_cd_head.num_classes
        self.out_channels = self.decode_head.binary_cd_head.out_channels

        self.semantic_num_classes = self.decode_head.semantic_cd_head.num_classes
        self.semantic_out_channels = self.decode_head.semantic_cd_head.out_channels

        self.align_corners = {
            'seg_logits': self.decode_head.binary_cd_head.align_corners,
            'seg_logits_from': self.decode_head.semantic_cd_head.align_corners,
            'seg_logits_to': self.decode_head.semantic_cd_head_aux.align_corners}
        self.thresholds = {
            'seg_logits': self.decode_head.binary_cd_head.threshold,
            'seg_logits_from': self.decode_head.semantic_cd_head.threshold,
            'seg_logits_to': self.decode_head.semantic_cd_head_aux.threshold}

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:

        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        feat_from = self.backbone(img_from)
        feat_to = self.backbone(img_to)
        if self.with_neck:
            feat_from = self.neck(feat_from)
            feat_to = self.neck(feat_to)
        x = (feat_from, feat_to)

        return x

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        semantic_channels = self.semantic_out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = dict(
            seg_logits=inputs.new_zeros((batch_size, out_channels, h_img, w_img)),
            seg_logits_from=inputs.new_zeros((batch_size, semantic_channels, h_img, w_img)),
            seg_logits_to=inputs.new_zeros((batch_size, semantic_channels, h_img, w_img))
        )
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]

                crop_seg_logits = self.encode_decode(crop_img, batch_img_metas)
                for seg_name, crop_seg_logit in crop_seg_logits.items():
                    preds[seg_name] += F.pad(crop_seg_logit,
                                (int(x1), int(preds[seg_name].shape[3] - x2), int(y1),
                                    int(preds[seg_name].shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        for seg_name, pred in preds.items():
            preds[seg_name] = pred / count_mat

        return preds

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:

        C = dict()
        for seg_name, seg_logit in seg_logits.items():
            batch_size, _C, H, W = seg_logit.shape
            C[seg_name] = _C

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            for seg_name, seg_logit in seg_logits.items():
                if not only_prediction:
                    img_meta = data_samples[i].metainfo

                    if 'img_padding_size' not in img_meta:
                        padding_size = img_meta.get('padding_size', [0] * 4)
                    else:
                        padding_size = img_meta['img_padding_size']
                    padding_left, padding_right, padding_top, padding_bottom =\
                        padding_size

                    i_seg_logit = seg_logit[i:i + 1, :,
                                            padding_top:H - padding_bottom,
                                            padding_left:W - padding_right]

                    flip = img_meta.get('flip', None)
                    if flip:
                        flip_direction = img_meta.get('flip_direction', None)
                        assert flip_direction in ['horizontal', 'vertical']
                        if flip_direction == 'horizontal':
                            i_seg_logit = i_seg_logit.flip(dims=(3, ))
                        else:
                            i_seg_logit = i_seg_logit.flip(dims=(2, ))

                    i_seg_logit = resize(
                        i_seg_logit,
                        size=img_meta['ori_shape'],
                        mode='bilinear',
                        align_corners=self.align_corners[seg_name],
                        warning=False).squeeze(0)
                else:
                    i_seg_logit = seg_logit[i]

                if C[seg_name] > 1:
                    i_seg_pred = i_seg_logit.argmax(dim=0, keepdim=True)
                else:
                    i_seg_logit = i_seg_logit.sigmoid()
                    i_seg_pred = (i_seg_logit >
                                    self.thresholds[seg_name]).to(i_seg_logit)

                pred_name = '_' + seg_name.split('_')[-1]\
                    if seg_name.split('_')[-1] in ['from', 'to'] else ''
                pred_name = 'pred_sem_seg' + pred_name
                data_samples[i].set_data({
                    seg_name:
                    PixelData(**{'data': i_seg_logit}),
                    pred_name:
                    PixelData(**{'data': i_seg_pred})
                })

        if self.postprocess_pred_and_label is not None:
            if self.postprocess_pred_and_label == 'cover_semantic':
                for data_sample in data_samples:

                    data_sample.pred_sem_seg_from.data = data_sample.pred_sem_seg_from.data + 1
                    data_sample.pred_sem_seg_to.data = data_sample.pred_sem_seg_to.data + 1
                    data_sample.pred_sem_seg_from.data = data_sample.pred_sem_seg_from.data *\
                                                            data_sample.pred_sem_seg.data
                    data_sample.pred_sem_seg_to.data = data_sample.pred_sem_seg_to.data *\
                                                            data_sample.pred_sem_seg.data

                    data_sample.gt_sem_seg_from.data[data_sample.gt_sem_seg_from.data == 255] = -1
                    data_sample.gt_sem_seg_from.data = data_sample.gt_sem_seg_from.data + 1
                    data_sample.gt_sem_seg_to.data[data_sample.gt_sem_seg_to.data == 255] = -1
                    data_sample.gt_sem_seg_to.data = data_sample.gt_sem_seg_to.data + 1
            else:
                raise ValueError(
                        f'`postprocess_pred_and_label` should be `cover_semantic` or None.')

        return data_samples
