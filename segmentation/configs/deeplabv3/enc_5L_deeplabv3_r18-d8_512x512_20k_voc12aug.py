_base_ = "./full_deeplabv3_r18-d8_512x512_20k_voc12aug.py"

freeze_layers = [
    "backbone", "decode_head", "~decode_head.conv_seg",
    "~decode_head.bottleneck", "~decode_head.aspp_modules"
]
