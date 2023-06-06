# Config naming format:
# [Type]_[Model]_[Resolution]_[Training Steps]_[Target Dataset].py
# Take config "enc_5L_deeplabv3_mv2_512x512_20k_voc12aug.py"(configs/deeplabv3mv2/enc_5L_deeplabv3_mv2_512x512_20k_voc12aug.py) as an example
# - enc_5L: training only the last 5 convolution layers; Possible options are:
#   - full: training the whold model
#   - enc_5L/10L: training the last 5/10 convolution layers with vanilla BP algorithm
#   - filt_5L/10L: training the last 5/10 convolution layers with our gradient filter
# - deeplabv3_mv2: DeepLabV3 with MobileNet-V2 backbone is used here
# - 512x512: The resolution of training images are 512x512
# - 20k: Train the model for 20k batches
# - voc12aug: Train on VOC12Aug dataset
# Commonly used args:
# --load-from <ckpt path> Load pretrained/calibrated model from <ckpt path>
# --cfg-options <options> Overwrite options in the config (e.g., --cfg-options data.samples_per_gpu=8 overwrites the number of samples per gpu to 8, so if you have one GPU, the batch size is 8; if you have two, the batch size is 16)
# --log-postfix <postfix> Add postfix to the name log file

# Launch with single GPU
# Configuration file is written for 4 GPUs, which means a 8 batch size (4 x samples_per_gpu(2) ). Thus to make sure we are running with the correct batch size, set samples_per_gpu to 8 for single GPU.

# Cityscapes->VOC12Aug, Train all layers in DeepLabV3-ResNet18 with vanilla BP 
python train.py configs/deeplabv3/full_deeplabv3_r18-d8_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8
# Cityscapes->VOC12Aug, Train last 5 layers in DeepLabV3-ResNet18 with vanilla BP 
python train.py configs/deeplabv3/enc_5L_deeplabv3_r18-d8_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8
# Cityscapes->VOC12Aug, Train last 10 layers in DeepLabV3-ResNet18 with vanilla BP
python train.py configs/deeplabv3/enc_10L_deeplabv3_r18-d8_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8
# Cityscapes->VOC12Aug, Train last 5 layers in DeepLabV3-ResNet18 with gradient filter
python train.py configs/deeplabv3/filt_5L_deeplabv3_r18-d8_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8
# Cityscapes->VOC12Aug, Train last 10 layers in DeepLabV3-ResNet18 with gradient filter
python train.py configs/deeplabv3/filt_10L_deeplabv3_r18-d8_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8

# Launch with multiple GPUs
# Change the second argument "4" to the number of GPUs
# Append arg --cfg-options data.samples_per_gpu=(8/#gpus) if you are using a different number of GPUs.
./dist_train.sh configs/deeplabv3/filt_5L_deeplabv3_r18-d8_512x512_20k_voc12aug.py 4 --load-from calib/calib_deeplabv3_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth
