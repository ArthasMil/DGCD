
dataset_type = 'SECOND_Dataset'
data_root = 'Second'

crop_size = (512, 512)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),

    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=(512, 512), keep_ratio=True),

    dict(type='MultiImgMultiAnnLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
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
            img_path_from='train/A/',
            img_path_to='train/B/',
            seg_map_path='train/label_bn_vis/',
            seg_map_path_from='train/labelA/',
            seg_map_path_to='train/labelB/'),
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
            img_path_from='test/A/',
            img_path_to='test/B/',
            seg_map_path='test/label_bn_vis/',
            seg_map_path_from='test/labelA/',
            seg_map_path_to='test/labelB/'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='test/A/',
            img_path_to='test/B/',
            seg_map_path='test/label_bn_vis/',
            seg_map_path_from='test/labelA/',
            seg_map_path_to='test/labelB/'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type='SCDMetric',
    iou_metrics=['mFscore', 'mIoU'],
    cal_sek=True)
test_evaluator = val_evaluator
