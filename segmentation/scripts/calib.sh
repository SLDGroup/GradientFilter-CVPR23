# Calibration
# You can skip this step by using provided calibrated checkpoint

# Download mmsegmentation pretrained ckpts
#   DeepLabV3-ResNet18 pretrained on Cityscapes
mim download mmsegmentation --config deeplabv3_r18-d8_512x1024_80k_cityscapes --dest pretrained_ckpts
#   DeepLabV3-MobileNetV2 pretrained on Cityscapes
mim download mmsegmentation --config deeplabv3_m-v2-d8_512x1024_80k_cityscapes --dest pretrained_ckpts
#   DeepLabV3-MobileNetV2 pretrained on ADE20K
mim download mmsegmentation --config deeplabv3_m-v2-d8_512x512_160k_ade20k --dest pretrained_ckpts
#   FCN-ResNet18 pretrained on Cityscapes
mim download mmsegmentation --config fcn_r18-d8_512x1024_80k_cityscapes --dest pretrained_ckpts
#   PSPNet-ResNet18 pretrained on Cityscapes
mim download mmsegmentation --config pspnet_r18-d8_512x1024_80k_cityscapes --dest pretrained_ckpts
#   PSPNet-MobileNetV2 pretrained on Cityscapes
mim download mmsegmentation --config pspnet_m-v2-d8_512x1024_80k_cityscapes --dest pretrained_ckpts
#   PSPNet-MobileNetV2 pretrained on ADE20K
mim download mmsegmentation --config pspnet_m-v2-d8_512x512_160k_ade20k --dest pretrained_ckpts
#   UPerNet-ResNet18 pretrained on Cityscapes
mim download mmsegmentation --config upernet_r18_512x1024_40k_cityscapes --dest pretrained_ckpts
#   UPerNet-ResNet18 pretrained on ADE20K
mim download mmsegmentation --config upernet_r18_512x512_80k_ade20k --dest pretrained_ckpts

# Calibrate the pretrained model on the target dataset
# DeepLabV3-ResNet18 Cityscapes -> VOC12Aug
python train.py configs/deeplabv3/calib_deeplabv3_r18-d8_512x512_1k_voc12aug.py --load-from pretrained_ckpts/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth --cfg-options data.samples_per_gpu=8 --log-postfix cityscapes
# DeepLabV3-MobileNetV2 Cityscapes -> VOC12Aug (More training steps since the whole decoder is changed)
python train.py configs/deeplabv3mv2/calib_deeplabv3_mv2_512x512_5k_voc12aug.py --load-from pretrained_ckpts/deeplabv3_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-bef03590.pth --cfg-options data.samples_per_gpu=8 --log-postfix cityscapes
# DeepLabV3-MobileNetV2 ADE20K -> VOC12Aug
python train.py configs/deeplabv3mv2/calib_deeplabv3_mv2_512x512_5k_voc12aug.py --load-from pretrained_ckpts/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth --cfg-options data.samples_per_gpu=8 --log-postfix ade20k
# DeepLabV3-MobileNetV2 ADE20K -> Cityscapes
python train.py configs/deeplabv3mv2/calib_deeplabv3_mv2_512x1024_5k_cityscapes.py --load-from pretrained_ckpts/deeplabv3_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-bef03590.pth --cfg-options data.samples_per_gpu=8 --log-postfix ade20k
# FCN-ResNet18 Cityscapes -> VOC12Aug
python train.py configs/fcn/calib_fcn_r18-d8_512x512_1k_voc12aug.py --load-from pretrained_ckpts/fcn_r18-d8_512x1024_80k_cityscapes_20201225_021327-6c50f8b4.pth --cfg-options data.samples_per_gpu=8 --log-postfix cityscapes
# PSPNet-ResNet18 Cityscapes -> VOC12Aug
python train.py configs/pspnet/calib_pspnet_r18-d8_512x512_1k_voc12aug.py --load-from pretrained_ckpts/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth --cfg-options data.samples_per_gpu=8 --log-postfix cityscapes
# PSPNet-MobileNetV2 Cityscapes -> VOC12Aug
python train.py configs/pspnetmv2/calib_pspnet_mv2-d8_512x512_5k_voc12aug.py --load-from pretrained_ckpts/pspnet_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-19e81d51.pth --cfg-options data.samples_per_gpu=8 --log-postfix cityscapes
# PSPNet-MobileNetV2 ADE20K -> VOC12Aug
python train.py configs/pspnetmv2/calib_pspnet_mv2-d8_512x512_5k_voc12aug.py --load-from pretrained_ckpts/pspnet_m-v2-d8_512x512_160k_ade20k_20200825_214953-f5942f7a.pth --cfg-options data.samples_per_gpu=8 --log-postfix ade20k
# PSPNet-MobileNetV2 ADE20K -> Cityscapes
python train.py configs/pspnetmv2/calib_pspnet_mv2-d8_512x1024_5k_cityscapes.py --load-from pretrained_ckpts/pspnet_m-v2-d8_512x512_160k_ade20k_20200825_214953-f5942f7a.pth --cfg-options data.samples_per_gpu=8 --log-postfix ade20k
# UPerNet-ResNet18 Cityscapes -> VOC12Aug
python train.py configs/upernet/calib_upernet_r18_512x512_1k_voc12aug.py --load-from pretrained_ckpts/upernet_r18_512x1024_40k_cityscapes_20220615_113231-12ee861d.pth --cfg-options data.samples_per_gpu=8 --log-postfix cityscapes
# UPerNet-ResNet18 ADE20K -> VOC12Aug
python train.py configs/upernet/calib_upernet_r18_512x512_1k_voc12aug.py --load-from pretrained_ckpts/upernet_r18_512x512_80k_ade20k_20220614_110319-22e81719.pth --cfg-options data.samples_per_gpu=8 --log-postfix ade20k
# UPerNet-ResNet18 ADE20K -> Cityscapes
python train.py configs/upernet/calib_upernet_r18_512x1024_1k_cityscapes.py --load-from pretrained_ckpts/upernet_r18_512x512_80k_ade20k_20220614_110319-22e81719.pth --cfg-options data.samples_per_gpu=8 --log-postfix ade20k