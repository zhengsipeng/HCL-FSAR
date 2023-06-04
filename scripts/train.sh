#!/usr/bin/env bash
set -x

python train.py \
    --model_name HCL_SSv2_OTAM_100 \
    --dataset ssv2_100_otam \
    --num_epochs 15 \
    --lr_drop 10 \
    --scheduler_gamma 0.1 \
    --optimizer adamw \
    --learning_rate 1e-4 \
    --num_train_episode 1000 \
    --setname train_val \
    --method hcl \
    --topK 40 \
    --sigma_global 1 \
    --sigma_temp 0.5 \
    --sigma_spa 0.5 \
    --batch_size 24 \
    --lr_scdler lr_drop \
    --data_aug type1 \
    --enc_layers 1 \
    --d_model 1152 \
    --num_gpus 4 \
    --amp_opt_level O0 