_base_ = 'full_fcn_r18-d8_512x512_20k_voc12aug.py'

freeze_layers = [
    "backbone",
    "~backbone.layer4.1",
]
gradient_filter = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.conv_cat", type='cbr', radius=4),
        dict(path="decode_head.convs.0", type='cbr', radius=4),
        dict(path="decode_head.convs.1", type='cbr', radius=4),
        dict(path="backbone.layer4.1", type='resnet_basic_block', radius=4),
    ]
)

