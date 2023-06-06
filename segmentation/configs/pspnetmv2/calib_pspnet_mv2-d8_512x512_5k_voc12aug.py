_base_ = [
    '../../mmsegmentation/configs/_base_/models/pspnet_r50-d8.py',
    '../../mmsegmentation/configs/_base_/datasets/pascal_voc12_aug.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../mmsegmentation/configs/_base_/schedules/schedule_10k.py'
]
model = dict(
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6)),
    decode_head=dict(
        in_channels=320,
        channels=128,
        num_classes=21,
    ),
    auxiliary_head=None)

data = dict(samples_per_gpu=2)

freeze_layers = ["backbone"]

runner = dict(max_iters=5000)
evaluation = dict(interval=5000)
