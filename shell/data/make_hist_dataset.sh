#!/bin/bash

HW=250
CT_name="CT.nii.gz" #поменять
MR_name="MRI.nii.gz" #поменять
data_dir="../" #поменять
out_dir="../" #поменять
log_dir="../logs/hist"

mkdir -p $out_dir $log_dir

for which in "train" "valid" "test"
do
    for plane in "axial"
    do
        for hist_type in "normal" "avg" "colin"
        do
            python -u brain_dataset_utils/generate_total_hist_global.py \
                --plane $plane \
                --hist_type $hist_type \
                --which_set $which \
                --height $HW \
                --width $HW \
                --pkl_name "$out_dir/MR_hist_global_${HW}_${which}_${plane}_${hist_type}.pkl" \
                --data_dir $data_dir \
                --data_csv "" \ #поменять
                --CT_name $CT_name \
                --MR_name $MR_name \
                > "$log_dir/MR_hist_global_${HW}_${which}_${plane}_${hist_type}.log"
        done
    done
done
