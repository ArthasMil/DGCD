_base_ = [
    '../_base_/models/siam_upernet_convnext_t_t_multimodals_contrasive_with_refine.py', '../_base_/datasets/loveda_u2r_1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    decode_head=dict(
        num_classes=7,
        ),
    auxiliary_head=dict(num_classes=7),

    test_cfg=dict(mode='whole'),
    )

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]

optimizer=dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='mmseg.LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2000,save_best='mIoU',max_keep_ckpts=5),
                     visualization=dict(type='CDVisualizationHook',\
                                        interval=1, draw_on_from_to_img=False))

visualizer = dict(type='CDLocalVisualizer', alpha=1.0)

train_dataloader = dict(
    batch_size=4,
    num_workers=4)
