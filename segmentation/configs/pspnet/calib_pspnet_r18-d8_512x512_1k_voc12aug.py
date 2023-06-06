_base_ = [
    '../../mmsegmentation/configs/_base_/models/pspnet_r50-d8.py',
    '../../mmsegmentation/configs/_base_/datasets/pascal_voc12_aug.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../mmsegmentation/configs/_base_/schedules/schedule_10k.py'
]
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=21,
    ),
    auxiliary_head=None)

data = dict(samples_per_gpu=2)

freeze_layers = [
    "backbone", "decode_head", "~decode_head.conv_seg"
]

runner = dict(max_iters=1000)
