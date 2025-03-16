#!/bin/bash

HW=250
CT_name="ct.nii.gz"
MR_name="mr.nii.gz"
data_dir="../../data"
out_dir="../../processed_data"
log_dir="../../logs/hist"

mkdir -p "$out_dir" "$log_dir"

for which in "train" "val" "test"
do
    for plane in "axial"
    do
        for hist_type in "normal" "avg" "colin"
        do
            python -u ../brain_dataset_utils/generate_total_hist_global.py \
                --plane "$plane" \
                --hist_type "$hist_type" \
                --which_set "$which" \
                --height "$HW" \
                --width "$HW" \
                --pkl_name "${out_dir}/MR_hist_global_${HW}_${which}_${plane}_${hist_type}.pkl" \
                --data_dir "${data_dir}/${which}" \
                --data_csv "" \
                --CT_name "$CT_name" \
                --MR_name "$MR_name" \
                > "${log_dir}/MR_hist_global_${HW}_${which}_${plane}_${hist_type}.log"
        done
    done
done