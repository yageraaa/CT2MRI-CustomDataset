#!/bin/bash
cd "$(dirname "$0")/.."

date="$(date +%y%m%d)"
config_name="BBDM_base.yaml"
HW=250
plane="axial"
gpu_ids="0"
batch=16 #поменять если поменял в конфиге
ddim_eta=0.0
dataset_type="ct2mr_aligned_global_hist_context"

exp_name="${date}_${HW}_BBDM_${plane}_DDIM"

results_dir="results/ct2mr_${HW}" #папка для резов
mkdir -p "$results_dir/$exp_name/checkpoint"

python -u main.py \
    --train \
    --exp_name $exp_name \
    --config "configs/$config_name" \
    --dataset_type $dataset_type \
    --HW $HW \
    --plane $plane \
    --batch $batch \
    --ddim_eta $ddim_eta \
    --sample_at_start \
    --save_top \
    --gpu_ids $gpu_ids
