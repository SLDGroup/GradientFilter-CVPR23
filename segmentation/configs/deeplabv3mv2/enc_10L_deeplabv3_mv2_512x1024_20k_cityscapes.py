_base_ = [
    'full_deeplabv3_mv2_512x1024_20k_cityscapes.py',
]

freeze_layers = [
    "backbone", "~backbone.layer7", "~backbone.layer6.2.conv.2"
]
