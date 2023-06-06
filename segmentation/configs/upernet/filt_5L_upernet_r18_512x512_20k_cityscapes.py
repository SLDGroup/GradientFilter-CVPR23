_base_ = 'full_upernet_r18_512x512_20k_cityscapes.py'

freeze_layers = [
    'backbone', 'decode_head', '~decode_head.conv_seg',
    '~decode_head.fpn_bottleneck', '~decode_head.fpn_convs', '~decode_head.bottleneck'
]

gradient_filter = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.fpn_bottleneck", type='cbr', radius=8),
        dict(path="decode_head.fpn_convs.0", type='cbr', radius=8),
        dict(path="decode_head.fpn_convs.1", type='cbr', radius=4),
        dict(path="decode_head.fpn_convs.2", type='cbr', radius=2),
        dict(path="decode_head.bottleneck", type='cbr', radius=2),
    ]
)
