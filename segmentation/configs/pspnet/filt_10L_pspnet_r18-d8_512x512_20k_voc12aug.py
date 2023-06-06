_base_ = "./full_pspnet_r18-d8_512x512_20k_voc12aug.py"

freeze_layers = [
    "backbone", "~backbone.layer4",
]


gradient_filter = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.bottleneck", type='cbr', radius=4),
        dict(path="backbone.layer4.1", type='resnet_basic_block', radius=4),
        dict(path="backbone.layer4.0", type='resnet_basic_block', radius=4),
    ]
)
