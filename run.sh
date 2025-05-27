#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=CRNP_BraTS_[80,160,160]_SGD_b2_lr-2

CUDA_VISIBLE_DEVICES=$1 python train_CRNP.py \
--snapshot_dir=snapshots/$name/ \
--input_size=80,160,160 \
--batch_size=2 \
--num_gpus=1 \
--num_steps=185000 \
--val_pred_every=2000 \
--learning_rate=1e-2 \
--num_classes=3 \
--num_workers=4 \
--train_list=BraTS20_train.csv \
--val_list=BraTS20_val.csv \
--random_mirror=True \
--random_scale=True \
--weight_std=True \
--train_only \
--reload_path=snapshots/model/final.pth \
--reload_from_checkpoint=False > logs/${time}_train_${name}.log 2>&1 &
