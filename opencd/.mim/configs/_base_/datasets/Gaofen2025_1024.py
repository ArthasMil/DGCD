
dataset_type = 'GaofenDataset2025'
data_root = 'GFData'

image_scale = (1024,1024)
crop_size = (1024, 1024)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),

    dict(type='MultiImgResize', scale=image_scale, keep_ratio=True),

    dict(type='MultiImgRandomRotFlip'),
    dict(type='MultiImgRandomFlip',direction='horizontal', prob=0.5),
    dict(type='MultiImgRandomFlip',direction='vertical', prob=0.5),
    dict(type='MultiImgRandomFlip',direction='diagonal', prob=0.5),
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=image_scale, keep_ratio=True),

    dict(type='MultiImgPackSegInputs')
]

val_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=image_scale, keep_ratio=True),

    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

img_ratios = [1.0]
tta_pipeline = [
    dict(type='MultiImgLoadImageFromFile', backend_args=None),

    dict(type='MultiImgResize', scale=image_scale, keep_ratio=True),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='MultiImgResize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
                dict(type='MultiImgRandomFlip', prob=1., direction='horizontal'),
                dict(type='MultiImgRandomFlip', prob=0., direction='vertical'),
                dict(type='MultiImgRandomFlip', prob=1., direction='vertical')
            ],

            [dict(type='MultiImgPackSegInputs')]
        ]
        )
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,

        data_prefix=dict(
            img_path_from='train/A',
            img_path_to='train/B',
            seg_map_path='train/label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='val/A',
            img_path_to='val/B',
            seg_map_path='val/label'),
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='test/A',
            img_path_to='test/B',

            ),
        pipeline=test_pipeline))

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(
    type='mmseg.IoUMetric',
    iou_metrics=['mFscore', 'mIoU'],
    format_only=True,
    output_dir='results')
