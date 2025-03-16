#!/bin/bash
cd "$(dirname "$0")/.."

config_name="BBDM_base.yaml"
HW=250
plane="axial"
ddim_eta=0.0
gpu_ids="0"
test_epoch="34"

exp_name="241213_250_BBDM_axial_DDIM_MR_global_hist_context"

resume_model="../results/ct2mr_${HW}/${exp_name}/checkpoint/latest_model_${test_epoch}.pth"
resume_optim="../results/ct2mr_${HW}/${exp_name}/checkpoint/latest_optim_sche_${test_epoch}.pth"

python main.py \
    --exp_name "$exp_name" \
    --config "configs/${config_name}" \
    --sample_to_eval \
    --gpu_ids "$gpu_ids" \
    --resume_model "$resume_model" \
    --resume_optim "$resume_optim" \
    --HW "$HW" \
    --plane "$plane" \
    --ddim_eta "$ddim_eta" \
    >> "../results/ct2mr_${HW}/${exp_name}/testing.log" 2>&1