
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
model = dict(
    type='SiamEncoderDecoder',
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
    neck=dict(
        type='FarSegFPN',
        policy='concat',
        in_channels=[128, 256, 1024, 1024],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='ChangeStarHead',
        inference_mode='t1t2',
        in_channels=[1, 1, 1, 1, 1],
        in_index=[0, 1, 2, 3, 4],
        channels=96,
        num_classes=2,
        out_channels=1,
        threshold=0.5,
        seg_head_cfg=dict(
            type='FarSegHead',
            in_channels=[256, 256, 256, 256, 512],
            in_index=[0, 1, 2, 3, 4],
            fsr_channels=256,
            channels=128,
            dropout_ratio=0.,
            norm_cfg=norm_cfg,
            align_corners=False),
        changemixin_cfg=dict(
            in_channels=128 * 2,
            inner_channels=96,
            num_convs=1,
            ),

         loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
