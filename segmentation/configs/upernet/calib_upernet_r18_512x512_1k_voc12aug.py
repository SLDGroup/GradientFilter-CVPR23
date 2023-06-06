_base_ = [
    '../../mmsegmentation/configs/_base_/models/upernet_r50.py',
    '../../mmsegmentation/configs/_base_/datasets/pascal_voc12_aug.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../mmsegmentation/configs/_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=21),
    auxiliary_head=None)

freeze_layers = [
    "backbone", "decode_head", "~decode_head.conv_seg", "~auxiliary_head.conv_seg"
]

data = dict(samples_per_gpu=2)

runner = dict(max_iters=1000)
