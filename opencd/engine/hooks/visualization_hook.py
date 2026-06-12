# Copyright (c) Open-CD. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmengine.runner import Runner

from mmseg.engine import SegVisualizationHook
from mmseg.structures import SegDataSample
from opencd.registry import HOOKS
from opencd.visualization import CDLocalVisualizer

@HOOKS.register_module()
class CDVisualizationHook(SegVisualizationHook):
    def __init__(self,
                 img_shape: tuple = None,
                 draw_on_from_to_img: bool = False,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self.img_shape = img_shape
        self.draw_on_from_to_img = draw_on_from_to_img
        if self.draw_on_from_to_img:
            warnings.warn('`draw_on_from_to_img` works only in '
                          'semantic change detection.')
        self._visualizer: CDLocalVisualizer =\
            CDLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:

            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        if self.draw is False or mode == 'train':
            return

        if self.every_n_inner_iters(batch_idx, self.interval):

            for output in outputs:
                img_path = output.img_path[0]
                img_from_to = []
                window_name = osp.basename(img_path).split('.')[0]
                if self.img_shape is not None:
                    assert len(self.img_shape) == 3,\
                        '`img_shape` should be (H, W, C)'
                else:
                    img_bytes = fileio.get(
                        img_path, backend_args=self.backend_args)
                    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                    self.img_shape = img.shape

                if self.draw_on_from_to_img:

                    for _img_path in output.img_path:
                        _img_bytes = fileio.get(
                            _img_path, backend_args=self.backend_args)
                        _img = mmcv.imfrombytes(_img_bytes, channel_order='rgb')
                        img_from_to.append(_img)

                img = np.zeros(self.img_shape)
                self._visualizer.add_datasample(
                    window_name,
                    img,
                    img_from_to,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter,
                    draw_gt=False)
