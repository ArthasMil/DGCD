checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'
crop_size = (
    1024,
    1024,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='DualInputSegDataPreProcessor')
data_root = 'Loveda_rural_Train+rural_Val_urban_Test'
dataset_type = 'LoveDADataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=5,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_on_from_to_img=False, interval=1, type='CDVisualizationHook'))
default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_scale = (
    1024,
    1024,
)
img_ratios = [
    1.0,
]

custom_hooks = [
    dict(type='mmseg.EMAHook',  priority='NORMAL')
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=896,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='mmseg.CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        num_convs=1,
        type='mmseg.FCNHead'),
    backbone=dict(
        arch='base',
        drop_path_rate=0.4,
        gap_before_final_norm=False,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth',
            prefix='backbone.',
            type='Pretrained'),
        layer_scale_init_value=1.0,
        out_indices=[
            0,
            1,
            2,
            3,
        ],
        type='mmpretrain.ConvNeXt'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=32),
        type='DualInputSegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        branch_channels=[
            [
                128,
                96,
            ],
            [
                256,
                192,
            ],
            [
                512,
                384,
            ],
            [
                1024,
                768,
            ],
        ],
        channels=512,

        contrast_losses=[
            dict(

                loss_weight=0.10,
                out_channels=128,
                temperature=0.07,
                type='CrossModalContrastiveLoss'),
            dict(

                loss_weight=0.10,
                out_channels=128,
                temperature=0.07,
                type='CrossModalContrastiveLoss'),
            dict(

                loss_weight=0.15,
                out_channels=128,
                temperature=0.07,
                type='CrossModalContrastiveLoss'),
            dict(

                loss_weight=0.15,
                out_channels=128,
                temperature=0.07,
                type='CrossModalContrastiveLoss'),
        ],
        dropout_ratio=0.1,
        in_channels=[
            224,
            448,
            896,
            1792,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='mmseg.CrossEntropyLoss', use_sigmoid=False),
        mask_refiner_cfg=dict(
            act_cfg=dict(inplace=True, type='ReLU'),
            norm_cfg=dict(type='BN'),
            refine_channels=128),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        refine_loss_weight=1.0,
        refine_scale=0,
        type='ContrastiveUPerHeadWithMaskRefine'),
    neck=dict(policy='concat', type='FeatureFusionNeck'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='SiamEncoderDecoderTwoBack')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    constructor='mmseg.LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12),
    type='AmpOptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=0.0001,
    momentum=0.9,
    type='AdamW',
    weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=800, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=800,
        by_epoch=False,
        end=20000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='test_upload/Rural/A',
            img_path_to='test_upload/Rural/B'),
        data_root='Loveda_rural_Train+rural_Val_urban_Test',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='MultiImgResize'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='LoveDADataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    format_only=True,
    iou_metrics=[
        'mFscore',
        'mIoU',
    ],
    output_dir='results',
    type='mmseg.IoUMetric')
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='MultiImgResize'),
    dict(type='MultiImgPackSegInputs'),
]
train_cfg = dict(max_iters=20000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path_from='test/A_1024',
            img_path_to='test/B_1024',
            seg_map_path='test/labels_1024'),
        data_root='Loveda_rural_Train+rural_Val_urban_Test',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(reduce_zero_label=True, type='MultiImgLoadAnnotations'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='MultiImgResize'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    1024,
                    1024,
                ),
                type='MultiImgRandomCrop'),
            dict(type='MultiImgRandomRotFlip'),
            dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
            dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
            dict(direction='diagonal', prob=0.5, type='MultiImgRandomFlip'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='LoveDADataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(reduce_zero_label=True, type='MultiImgLoadAnnotations'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='MultiImgResize'),
    dict(
        cat_max_ratio=0.75,
        crop_size=(
            1024,
            1024,
        ),
        type='MultiImgRandomCrop'),
    dict(type='MultiImgRandomRotFlip'),
    dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
    dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
    dict(direction='diagonal', prob=0.5, type='MultiImgRandomFlip'),
    dict(type='MultiImgPackSegInputs'),
]
tta_model = dict(type='mmseg.SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='MultiImgLoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='MultiImgResize'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=1.0, type='MultiImgResize'),
            ],
            [
                dict(
                    direction='horizontal',
                    prob=0.0,
                    type='MultiImgRandomFlip'),
                dict(
                    direction='horizontal',
                    prob=1.0,
                    type='MultiImgRandomFlip'),
                dict(
                    direction='vertical', prob=0.0, type='MultiImgRandomFlip'),
                dict(
                    direction='vertical', prob=1.0, type='MultiImgRandomFlip'),
            ],
            [
                dict(type='MultiImgPackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='train/A_1024',
            img_path_to='train/B_1024',
            seg_map_path='train/labels_1024'),
        data_root='Loveda_rural_Train+rural_Val_urban_Test',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='MultiImgResize'),
            dict(reduce_zero_label=True, type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='LoveDADataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
val_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='MultiImgResize'),
    dict(reduce_zero_label=True, type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
vis_backends = [
    dict(type='CDLocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
work_dir = 'myconfigs/logs/conv_b_s_Contrassive_Loss_u2r_1024_with_refine_dynamicWeights_20251202'
