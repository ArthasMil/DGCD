_base_ = [
    '../_base_/models/siam_upernet_swin_b_s_multimodals.py', '../_base_/datasets/Gaofen2025.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=200000, val_interval=1000)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1000,save_best='mIoU',max_keep_ckpts=5),visualization=dict(type='CDVisualizationHook',\
                                        interval=1, draw_on_from_to_img=False))

visualizer = dict(type='CDLocalVisualizer', alpha=1.0)

train_dataloader = dict(
    batch_size=20,
    num_workers=20)
