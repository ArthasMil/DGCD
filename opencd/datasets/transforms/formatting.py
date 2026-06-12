# Copyright (c) Open-CD. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.structures import SegDataSample
from opencd.registry import TRANSFORMS

@TRANSFORMS.register_module()
class MultiImgPackSegInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'seg_map_path_from',
                            'seg_map_path_to', 'ori_shape','img_shape',
                            'pad_shape', 'scale_factor', 'flip',
                            'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'img' in results:
            def _transform_img(img):
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if not img.flags.c_contiguous:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                else:
                    img = img.transpose(2, 0, 1)
                    img = to_tensor(img).contiguous()
                return img

            imgs = [_transform_img(img) for img in results['img']]
            imgs = torch.cat(imgs, axis=0)
            packed_results['inputs'] = imgs

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                data=to_tensor(results['gt_seg_map'][None,
                                                     ...].astype(np.int64)))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_seg_map_from' in results:
            gt_sem_seg_data_from = dict(
                data=to_tensor(results['gt_seg_map_from'][None,
                                                     ...].astype(np.int64)))
            data_sample.set_data(dict(gt_sem_seg_from=PixelData(**gt_sem_seg_data_from)))

        if 'gt_seg_map_to' in results:
            gt_sem_seg_data_to = dict(
                data=to_tensor(results['gt_seg_map_to'][None,
                                                     ...].astype(np.int64)))
            data_sample.set_data(dict(gt_sem_seg_to=PixelData(**gt_sem_seg_data_to)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
