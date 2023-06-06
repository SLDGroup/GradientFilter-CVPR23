_base_ = "./full_pspnet_mv2-d8_512x1024_20k_cityscapes.py"

freeze_layers = ["backbone"]

gradient_filter = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.bottleneck", type='cbr', radius=4),
    ]
)
