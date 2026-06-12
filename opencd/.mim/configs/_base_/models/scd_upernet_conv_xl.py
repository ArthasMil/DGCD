
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth'

model = dict(
    type='SiamEncoderMultiDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='xlarge',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        type='GeneralSCDHead',
        binary_cd_neck=dict(
            type='FeatureFusionNeck',
            policy='abs_diff'),
        binary_cd_head=dict(
            type='mmseg.UPerHead',
            in_channels=[v * 1 for v in [256, 512, 1024, 2048]],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=256,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        semantic_cd_head=dict(
            type='mmseg.UPerHead',
            in_channels=[256, 512, 1024, 2048],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=256,
            dropout_ratio=0.1,
            num_classes=6,
            ignore_index=255,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True))),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
