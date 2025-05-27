#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=Eval_CRNP_BraTS_[80,160,160]_SGD_b2_lr-2

CUDA_VISIBLE_DEVICES=$1 python eval_CRNP.py \
--input_size=80,160,160 \
--num_classes=3 \
--data_list=BraTS20_test.csv \
--weight_std=True \
--restore_from=snapshots/CRNP_BraTS_[80,160,160]_SGD_b2_lr-2/last.pth \
> logs/${time}_train_${name}.log 2>&1 &
