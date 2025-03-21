#!/bin/bash
cd "$(dirname "$0")/../.."
echo "Current directory: $(pwd)"

config_name="BBDM_base.yaml"
HW=250
plane="axial"
gpu_ids="0"
batch=1
ddim_eta=0.0
dataset_type="ct2mr_aligned_global_hist_context"

exp_name="$(date +%y%m%d)_${HW}_BBDM_${plane}_DDIM_MR_global_hist_context"

results_dir="./results/ct2mr_${HW}"
mkdir -p "$results_dir/$exp_name/checkpoint"

python -u ./main.py \
    --train \
    --exp_name "$exp_name" \
    --config "configs/$config_name" \
    --dataset_type "$dataset_type" \
    --HW "$HW" \
    --plane "$plane" \
    --batch "$batch" \
    --ddim_eta "$ddim_eta" \
    --sample_at_start \
    --save_top \
    --gpu_ids "$gpu_ids" \
    >> "$results_dir/$exp_name/training.log" 2>&1