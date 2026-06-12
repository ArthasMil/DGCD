
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth'

model = dict(
    type='SiamEncoderDecoderTwoBack',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmseg.SwinTransformer',
        pretrain_img_size=224,
        embed_dims=128,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    neck=dict(type='FeatureFusionNeck', policy='concat'),

    decode_head=dict(
        type='ContrastiveUPerHeadWithMaskRefine',

        in_channels=[224,448,896,1792],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),

        branch_channels=[
            [128, 96],
            [256, 192],
            [512, 384],
            [1024, 768]
        ],
        mask_refiner_cfg=dict(
        refine_channels=128,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),),
        refine_scale=0,
        refine_loss_weight=1.0,

        contrast_losses=[
            dict(
                type='CrossModalContrastiveLoss',
                out_channels=128,
                temperature=0.07,
                loss_weight=0.02

            ),
            dict(
                type='CrossModalContrastiveLoss',
                out_channels=128,
                temperature=0.07,
                loss_weight=0.02

            ),
            dict(
                type='CrossModalContrastiveLoss',
                out_channels=128,
                temperature=0.07,
                loss_weight=0.03

            ),
            dict(
                type='CrossModalContrastiveLoss',
                out_channels=128,
                temperature=0.07,
                loss_weight=0.03

            )
        ],

        ),
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
    test_cfg=dict(mode='whole'))
