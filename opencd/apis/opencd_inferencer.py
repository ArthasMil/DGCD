# Copyright (c) Open-CD. All rights reserved.
import os.path as osp
from typing import List, Optional, Union

import mmcv
import mmengine
import numpy as np

from mmcv.transforms import Compose

from mmseg.utils import ConfigType
from mmseg.apis import MMSegInferencer

class OpenCDInferencer(MMSegInferencer):

    def __init__(self,
                 classes: Optional[Union[str, List]] = None,
                 palette: Optional[Union[str, List]] = None,
                 dataset_name: Optional[str] = None,
                 scope: Optional[str] = 'opencd',
                 **kwargs) -> None:
        super().__init__(scope=scope, **kwargs)

        classes = classes if classes else self.model.dataset_meta.classes
        palette = palette if palette else self.model.dataset_meta.palette
        self.visualizer.set_dataset_meta(classes, palette, dataset_name)

    def _inputs_to_list(self, inputs: Union[str, np.ndarray]) -> list:
        return list(inputs)

    def visualize(self,
                  inputs: list,
                  preds: List[dict],
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  img_out_dir: str = '',
                  opacity: float = 1.0) -> List[np.ndarray]:
        if not show and img_out_dir == '' and not return_vis:
            return None
        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        self.visualizer.alpha = opacity

        results = []

        for single_inputs, pred in zip(inputs, preds):
            img_from_to = []
            for single_input in single_inputs:
                if isinstance(single_input, str):
                    img_bytes = mmengine.fileio.get(single_input)
                    img = mmcv.imfrombytes(img_bytes)
                    img = img[:, :, ::-1]
                    img_name = osp.basename(single_input)
                elif isinstance(single_input, np.ndarray):
                    img = single_input.copy()
                    img_num = str(self.num_visualized_imgs).zfill(8) + '_vis'
                    img_name = f'{img_num}.jpg'
                else:
                    raise ValueError('Unsupported input type:'
                                    f'{type(single_input)}')
                img_shape = img.shape
                img_from_to.append(img)

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != ''\
                else None

            img_zero_board = np.zeros(img_shape)
            self.visualizer.add_datasample(
                img_name,
                img_zero_board,
                img_from_to,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=True,
                out_file=out_file)
            if return_vis:
                results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results if return_vis else None

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        for transform in ('MultiImgLoadAnnotations', 'MultiImgLoadDepthAnnotation'):
            idx = self._get_transform_idx(pipeline_cfg, transform)
            if idx != -1:
                del pipeline_cfg[idx]

        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'MultiImgLoadImageFromFile')
        if load_img_idx == -1:
            raise ValueError(
                'MultiImgLoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'MultiImgLoadInferencerLoader'
        return Compose(pipeline_cfg)
