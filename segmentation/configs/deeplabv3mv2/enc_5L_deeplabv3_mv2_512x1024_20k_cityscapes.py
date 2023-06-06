_base_ = [
    'full_deeplabv3_mv2_512x1024_20k_cityscapes.py',
]

freeze_layers = [
    "backbone", "decode_head", "~decode_head.conv_seg",
    "~decode_head.bottleneck", "~decode_head.aspp_modules"
]
