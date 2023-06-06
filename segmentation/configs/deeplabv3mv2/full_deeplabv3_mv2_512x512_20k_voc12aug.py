_base_ = [
    '../../mmsegmentation/configs/_base_/models/deeplabv3_r50-d8.py',
    '../../mmsegmentation/configs/_base_/datasets/pascal_voc12_aug.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../mmsegmentation/configs/_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        norm_eval=True,
        # norm_cfg=dict(type='BN', requires_grad=False)
    ),
    decode_head=dict(
        in_channels=320,
        channels=128,
        num_classes=21,
        norm_eval=True,
        # norm_cfg=dict(requires_grad=False),
        dropout_ratio=0.0
    ),
    auxiliary_head=None)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ])
lr_config = dict(_delete_=True, policy='CosineAnnealing',
                 min_lr=1e-4, min_lr_ratio=None,
                 warmup='linear', warmup_iters=1000, warmup_ratio=1.0 / 10,
                 by_epoch=False)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

data = dict(
    samples_per_gpu=2,
)
