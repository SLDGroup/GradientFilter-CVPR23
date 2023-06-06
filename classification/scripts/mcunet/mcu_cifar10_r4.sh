#!/bin/bash

pwd
date

usr_group_kl=15.29
load_args="--model.load pretrained_ckpts/mcu/pretrain_15.29_cifar10/version_0/checkpoints/epoch=49-val-acc=0.950.ckpt"
logger_args="--logger.save_dir runs/cls/mcunet/cifar10"
data_args="--data.name cifar10 --data.data_dir data/cifar10 --data.partition 1 --data.usr_group data/cifar10/usr_group_${usr_group_kl}.npy"
trainer_args="--trainer.max_epochs 50"
model_args="--model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes 10 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
common_args="$trainer_args $data_args $model_args $load_args $logger_args"

echo $common_args


python trainer_cls.py --config configs/cls/mcunet/filt_last1_r4.yaml ${common_args} --logger.exp_name filt_l1_r4_${usr_group_kl} --model.with_grad_filter True
python trainer_cls.py --config configs/cls/mcunet/filt_last2_r4.yaml ${common_args} --logger.exp_name filt_l2_r4_${usr_group_kl} --model.with_grad_filter True
python trainer_cls.py --config configs/cls/mcunet/filt_last3_r4.yaml ${common_args} --logger.exp_name filt_l3_r4_${usr_group_kl} --model.with_grad_filter True
python trainer_cls.py --config configs/cls/mcunet/filt_last4_r4.yaml ${common_args} --logger.exp_name filt_l4_r4_${usr_group_kl} --model.with_grad_filter True

wait





