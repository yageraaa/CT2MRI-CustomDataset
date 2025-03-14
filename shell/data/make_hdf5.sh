#!/bin/bash

HW=250
CT_name="CT.nii.gz" #поменятье
MR_name="MRI.nii.gz" #поменять
data_dir="../" #поменять
out_dir="../" #поменять
log_dir="../logs/hdf5"

mkdir -p $out_dir $log_dir

for which in "train" "valid" "test"
do
    for plane in "axial"
    do
        python -u brain_dataset_utils/generate_total_hdf5_csv.py \
            --plane $plane \
            --which_set $which \
            --height $HW \
            --width $HW \
            --hdf5_name "$out_dir/${HW}_${which}_${plane}.hdf5" \
            --data_dir $data_dir \
            --data_csv "" \ #поменять если есть csv
            --CT_name $CT_name \
            --MR_name $MR_name \
            > "$log_dir/${HW}_${which}_${plane}.log"
    done
done

