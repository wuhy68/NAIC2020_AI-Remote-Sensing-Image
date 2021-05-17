# dataset settings
dataset_type = 'MyCustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsOne'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=None),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='RandomUpDownFlip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='SingleScaleFourFlipAug',
        img_scale=(512, 512),
        flip_lr=False,
        flip_ud=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='RandomUpDownFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='data/train_sub.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='data/eval_5000.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/test_B.json',
        pipeline=test_pipeline))
