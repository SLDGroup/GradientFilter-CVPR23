_base_ = 'full_upernet_r18_512x512_20k_cityscapes.py'

freeze_layers = [
    'backbone', 'decode_head', '~decode_head.conv_seg',
    '~decode_head.fpn_bottleneck', '~decode_head.fpn_convs', '~decode_head.bottleneck',
    '~decode_head.lateral_convs', '~decode_head.psp_modules.0', '~decode_head.psp_modules.1', 
]
