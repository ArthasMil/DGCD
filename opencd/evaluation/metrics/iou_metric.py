# Copyright (c) Open-CD. All rights reserved.
from typing import Optional

from mmseg.evaluation import IoUMetric

from opencd.registry import METRICS

@METRICS.register_module()
class CDIoUMetric(IoUMetric):

    @staticmethod
    def _normalize_img_path(data_sample) -> Optional[str]:
        if isinstance(data_sample, dict):
            img_path = data_sample.get('img_path', None)
        else:
            img_path = getattr(data_sample, 'img_path', None)
            if img_path is None and hasattr(data_sample, 'get'):
                img_path = data_sample.get('img_path', None)
        if isinstance(img_path, (list, tuple)):
            img_path = img_path[0]
        return img_path

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            img_path = self._normalize_img_path(data_sample)
            if img_path is None:
                continue
            if isinstance(data_sample, dict):
                data_sample['img_path'] = img_path
            else:
                data_sample.set_metainfo(dict(img_path=img_path))
        super().process(data_batch, data_samples)
