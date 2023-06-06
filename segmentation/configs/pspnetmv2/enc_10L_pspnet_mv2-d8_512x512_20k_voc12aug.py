_base_ = "./full_pspnet_mv2-d8_512x512_20k_voc12aug.py"

freeze_layers = [
    "backbone", "~backbone.layer7",
    "~backbone.layer6.2.conv.2", "~backbone.layer6.2.conv.1"
]
