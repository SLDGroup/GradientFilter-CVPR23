_base_ = "./full_pspnet_mv2-d8_512x1024_20k_cityscapes.py"

freeze_layers = [
    "backbone", "~backbone.layer7",
    "~backbone.layer6.2.conv.2", "~backbone.layer6.2.conv.1"
]
