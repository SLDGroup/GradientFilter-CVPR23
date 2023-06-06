_base_ = [
    '../../mmsegmentation/configs/_base_/models/pspnet_r50-d8.py',
    '../../mmsegmentation/configs/_base_/datasets/pascal_voc12_aug.py',
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
        in_channels=512,
        channels=128,
        norm_eval=True,
        # norm_cfg=dict(requires_grad=False),
        num_classes=21,
        dropout_ratio=0.0
    ),
    auxiliary_head=None)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ])

data = dict(samples_per_gpu=2)
lr_config = dict(_delete_=True, policy='CosineAnnealing',
                 min_lr=1e-4, min_lr_ratio=None,
                 warmup='linear', warmup_iters=1000, warmup_ratio=1.0 / 10,
                 by_epoch=False)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

data = dict(
    samples_per_gpu=2,
)
