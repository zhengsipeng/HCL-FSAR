#!/usr/bin/env bash
set -x

CUDA_VISIBLE_DEVICES=1,0,2,3 python eval.py \
    --ckpt_path models/hcl_ssv2_100_otam.pth \
    --dataset ssv2_100_otam \
    --method hcl \
    --sigma_global 1 \
    --sigma_temp 0.5 \
    --sigma_spa 0.5 \
    --data_aug type1 \
    --enc_layers 1 \
    --d_model 1152 \
    --num_gpus 4 \
    --amp_opt_level O0 \
    --shot 5
