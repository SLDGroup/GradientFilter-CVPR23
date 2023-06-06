_base_ = [
    'full_deeplabv3_mv2_512x512_20k_voc12aug.py',
]

freeze_layers = [
    "backbone", "~backbone.layer7", "~backbone.layer6.2.conv.2"
]

gradient_filter = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.bottleneck", type='cbr', radius=3),
        dict(path="decode_head.aspp_modules.0", type='cbr', radius=3),
        dict(path="decode_head.aspp_modules.1", type='cbr', radius=3),
        dict(path="decode_head.aspp_modules.2", type='cbr', radius=6),
        dict(path="decode_head.aspp_modules.3", type='cbr', radius=9),
        dict(path="backbone.layer7.0.conv.2", type='cbr', radius=3),
        dict(path="backbone.layer7.0.conv.1", type='cbr', radius=3),
        dict(path="backbone.layer7.0.conv.0", type='cbr', radius=3),
        dict(path="backbone.layer6.2.conv.2", type='cbr', radius=3),
    ]
)
