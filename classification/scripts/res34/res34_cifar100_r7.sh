#!/bin/bash

pwd
date

usr_group_kl=15.82
load_args="--model.load pretrained_ckpts/res34/pretrain_15.82_cifar100/version_0/checkpoints/epoch=08-val-acc=0.835.ckpt"
logger_args="--logger.save_dir runs/cls/res34/cifar100"
data_args="--data.name cifar100 --data.data_dir data/cifar100 --data.partition 1 --data.usr_group data/cifar100/usr_group_${usr_group_kl}.npy --data.train_workers 24 --data.val_workers 24"
trainer_args="--trainer.max_epochs 50"
model_args="--model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes 100 --model.momentum 0 --trainer.gradient_clip_val 2.0"
common_args="$trainer_args $data_args $model_args $load_args $logger_args"

echo $common_args

python trainer_cls.py --config configs/cls/res34/filt_last1_r7.yaml ${common_args} --logger.exp_name filt_l1_r7_${usr_group_kl} --model.with_grad_filter True
python trainer_cls.py --config configs/cls/res34/filt_last2_r7.yaml ${common_args} --logger.exp_name filt_l2_r7_${usr_group_kl} --model.with_grad_filter True
python trainer_cls.py --config configs/cls/res34/filt_last3_r7.yaml ${common_args} --logger.exp_name filt_l3_r7_${usr_group_kl} --model.with_grad_filter True
python trainer_cls.py --config configs/cls/res34/filt_last4_r7.yaml ${common_args} --logger.exp_name filt_l4_r7_${usr_group_kl} --model.with_grad_filter True

wait


