
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
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'

num_classes = 7
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
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(type='FeatureFusionNeck', policy='concat'),

    decode_head=dict(
    type='DepthGuidedMask2FormerHead',

    in_channels=[128, 256, 512, 1024],
    point_sample_num=4096,
    strides=[4, 8, 16, 32],
    feat_channels=128,
    out_channels=128,
    num_classes=num_classes,
    num_queries=50,
    num_transformer_feat_level=3,
    align_corners=False,

    branch_channels=[
        [128, 64],
        [256, 128],
        [512, 256],
        [1024, 512],
    ],
    refine_scale=0,

    contrastive_loss_cfg=dict(
        out_channels=128,
        temperature=0.07,
        loss_weight=0.15,
        downsample_ratio=4,
        initial_fusion_weight=0.3,
    ),

    mask_refiner_cfg=dict(
        refine_channels=32,
        norm_cfg=None,
        act_cfg=dict(type='ReLU', inplace=True),
    ),

    pixel_decoder=dict(
        type='mmdet.MSDeformAttnPixelDecoder',
        num_outs=3,
        norm_cfg=dict(type='GN', num_groups=16),
        act_cfg=dict(type='ReLU'),
        encoder=dict(
            num_layers=4,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=128,
                    num_heads=8,
                    num_levels=3,
                    num_points=3,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=True,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfg=dict(
                    embed_dims=128,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        positional_encoding=dict(
            num_feats=64, normalize=True),
        init_cfg=None),

    positional_encoding=dict(
        num_feats=64, normalize=True),

    transformer_decoder=dict(
        return_intermediate=True,
        num_layers=4,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=128,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                dropout_layer=None,
                batch_first=True),
            cross_attn_cfg=dict(
                embed_dims=128,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                dropout_layer=None,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=1024,
                num_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True),
                ffn_drop=0.0,
                dropout_layer=None,
                add_identity=True)),
        init_cfg=None),

    loss_cls=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.5,
        reduction='mean',
        class_weight=[1.0] * num_classes + [0.1]),
    loss_mask=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=5.0),
    loss_dice=dict(
        type='mmdet.DiceLoss',
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=5.0),
    train_cfg=dict(
        num_points=8192,
        oversample_ratio=2.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(type='mmdet.ClassificationCost', weight=1.5),
                dict(
                    type='mmdet.CrossEntropyLossCost',
                    weight=5.0,
                    use_sigmoid=True),
                dict(
                    type='mmdet.DiceCost',
                    weight=5.0,
                    pred_act=True,
                    eps=1.0)
            ]),
        sampler=dict(type='mmdet.MaskPseudoSampler'))),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )
