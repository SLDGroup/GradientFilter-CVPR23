#!/bin/bash

pwd
date

usr_group_kl=13.10

load_args="--model.load pretrained_ckpts/res34/pretrain_13.10_imagenet/version_0/checkpoints/epoch=155-val-acc=0.780.ckpt"

data_args="--data.name imagenet --data.data_dir data/imagenet --data.train_workers 32 --data.val_workers 32 --data.partition 1 --data.usr_group data/imagenet/usr_group_${usr_group_kl}.npy --data.batch_size 64"

model_args="--model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.lr_warmup 4 --model.num_classes 1000 --model.momentum 0.9 --model.anneling_steps 90 --model.scheduler_interval epoch "

trainer_args="--trainer.max_epochs 90 --trainer.gpus 4 --trainer.strategy ddp --trainer.gradient_clip_val 2.0 "

logger_args="--logger.save_dir runs/cls/res34/imagenet"

common_args="$trainer_args $data_args $model_args $logger_args $load_args"

python trainer_cls.py --config configs/cls/res34/filt_last2_r2.yaml ${common_args} --logger.exp_name filt_l2_${usr_group_kl} --model.with_grad_filter True
