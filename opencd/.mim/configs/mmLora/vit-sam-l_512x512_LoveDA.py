_base_ = [
    '../_base_/models/mm_vit-sam-l.py',
    '../_base_/datasets/loveda_r2u.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
    ]

crop_size = (512, 512)

model = dict(
    backbone=dict(
        encoder_cfg=dict(img_size=crop_size)),
    decode_head=dict(num_classes=7),
    test_cfg=dict(mode='whole')
    )

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.05))

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

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2000,save_best='mIoU',max_keep_ckpts=5),
                     visualization=dict(type='CDVisualizationHook',\
                                        interval=1, draw_on_from_to_img=False))

train_dataloader = dict(
    batch_size=6,
    num_workers=6)
