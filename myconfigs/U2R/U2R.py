default_scope = 'opencd'

data_root = 'Loveda_rural_Train+rural_Val_urban_Test'
image_scale = (1024, 1024)
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='CDVisualizationHook', draw_on_from_to_img=False, interval=1))

vis_backends = [dict(type='CDLocalVisBackend')]
visualizer = dict(
    type='CDLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=1.0)

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='SiamEncoderDecoderTwoBack',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth',
            prefix='backbone.')),
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        type='ContrastiveUPerHeadWithMaskRefine',
        in_channels=[224, 448, 896, 1792],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        branch_channels=[[128, 96], [256, 192], [512, 384], [1024, 768]],
        contrast_losses=[
            dict(type='CrossModalContrastiveLoss', out_channels=128, temperature=0.07, loss_weight=0.02),
            dict(type='CrossModalContrastiveLoss', out_channels=128, temperature=0.07, loss_weight=0.02),
            dict(type='CrossModalContrastiveLoss', out_channels=128, temperature=0.07, loss_weight=0.03),
            dict(type='CrossModalContrastiveLoss', out_channels=128, temperature=0.07, loss_weight=0.03),
        ],
        mask_refiner_cfg=dict(
            refine_channels=128,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True)),
        refine_scale=0,
        refine_loss_weight=1.0),
    auxiliary_head=dict(
        type='mmseg.FCNHead',
        in_channels=896,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=crop_size))

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations', reduce_zero_label=True),
    dict(type='MultiImgResize', scale=image_scale, keep_ratio=True),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomRotFlip'),
    dict(type='MultiImgRandomFlip', direction='horizontal', prob=0.5),
    dict(type='MultiImgRandomFlip', direction='vertical', prob=0.5),
    dict(type='MultiImgRandomFlip', direction='diagonal', prob=0.5),
    dict(type='MultiImgPackSegInputs'),
]

val_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=image_scale, keep_ratio=True),
    dict(type='MultiImgLoadAnnotations', reduce_zero_label=True),
    dict(type='MultiImgPackSegInputs'),
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=image_scale, keep_ratio=True),
    dict(type='MultiImgPackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='LoveDADataset',
        data_root=data_root,
        data_prefix=dict(
            img_path_from='test/A',
            img_path_to='test/B',
            seg_map_path='test/labels'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LoveDADataset',
        data_root=data_root,
        data_prefix=dict(
            img_path_from='train/A',
            img_path_to='train/B',
            seg_map_path='train/labels'),
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LoveDADataset',
        data_root=data_root,
        data_prefix=dict(
            img_path_from='test_upload/Rural/A',
            img_path_to='test_upload/Rural/B'),
        pipeline=test_pipeline))

val_evaluator = dict(type='CDIoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(
    type='CDIoUMetric',
    iou_metrics=['mFscore', 'mIoU'],
    format_only=True,
    output_dir='results')

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12),
    constructor='mmseg.LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', power=1.0, begin=1500, end=40000, eta_min=0.0, by_epoch=False),
]
