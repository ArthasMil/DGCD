_base_ = [
    '../_base_/models/siam_upernet_convnext_l_s_multimodals.py', '../_base_/datasets/Gaofen2025_1408_all.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(

    decode_head=dict(
    loss_decode=[
        dict(type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
        dict(type='mmseg.DiceLoss', use_sigmoid=False, loss_name='loss_dice', loss_weight=2.0)])

    )

custom_hooks = [
    dict(type='mmseg.EMAHook',  priority='NORMAL')
]

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

optimizer=dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=optimizer,
    paramwise_cfg = {
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12

            })

train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=1000)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1000,save_best='mIoU',max_keep_ckpts=5),
                     visualization=dict(type='CDVisualizationHook',\
                                        interval=1, draw_on_from_to_img=False))

visualizer = dict(type='CDLocalVisualizer', alpha=1.0)

train_dataloader = dict(
    batch_size=2,
    num_workers=2)

compile = True
