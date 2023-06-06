_base_ = [
    '../../mmsegmentation/configs/_base_/models/upernet_r50.py',
    '../../mmsegmentation/configs/_base_/datasets/cityscapes.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../mmsegmentation/configs/_base_/schedules/schedule_20k.py'
]

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        depth=18,
        norm_eval=True,
        # norm_cfg=dict(requires_grad=False)
    ),
    decode_head=dict(
        norm_eval=True,
        # norm_cfg=dict(requires_grad=False),
        dropout_ratio=0.0,
        in_channels=[64, 128, 256, 512],
        num_classes=19
    ),
    auxiliary_head=None)

data = dict(samples_per_gpu=2)

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
optimizer = dict(lr=0.01, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
