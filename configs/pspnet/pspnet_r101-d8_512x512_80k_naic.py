_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/naic2020_noaug.py',
    # '../_base_/datasets/gid5classes_noaug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(type='ResNetV1c', depth=101),
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8),
)

evaluation = dict(interval=8000, metric='mIoU')
# optimizer
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

work_dir = 'data/pspnet-r101_d16_1011'
