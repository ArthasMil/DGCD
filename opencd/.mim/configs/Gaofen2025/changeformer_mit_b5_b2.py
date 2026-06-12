_base_ = [
    '../_base_/models/changeformer_mit-b0_multimodal.py', '../_base_/datasets/Gaofen2025.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'

in_channel_mit_b5 = [64, 128, 320, 512]
in_channel_mit_b1 = [64, 128, 320, 512]
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[v * 2 for v in [64, 128, 320, 512]]))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
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
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1000,save_best='mIoU'),
                     visualization=dict(type='CDVisualizationHook',\
                                        interval=1, draw_on_from_to_img=False))

visualizer = dict(type='CDLocalVisualizer', alpha=1.0)

train_dataloader = dict(
    batch_size=18,
    num_workers=18)
