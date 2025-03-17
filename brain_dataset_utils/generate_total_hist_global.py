import argparse
import time
import h5py
import os
import numpy as np
import nibabel as nib
import pandas as pd
import pickle


def transpose_LPS_to_ITKSNAP_position(np_data, plane):
    if plane == 'axial':
        np_data = np.transpose(np_data, [1, 0, 2])
    elif plane == 'coronal':
        np_data = np.flip(np_data, axis=(2))
        np_data = np.transpose(np_data, [2, 0, 1])
    elif plane == 'sagittal':
        np_data = np.rot90(np_data, axes=(1, 2))
        np_data = np.transpose(np_data, [1, 2, 0])
    return np_data

def visualize_histogram(hist, num_bins, fig_out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_bins), hist, width=3)

    plt.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(fig_out_path), exist_ok=True)
    plt.savefig(fig_out_path, dpi=300)
    plt.close()

def create_hdf5_dataset(args):
    num_bins = 128
    epsilon = 0.0000001
    modal = 'MR'
    start_d = time.time()
    visualize = True
    scale = 10
    subject_dir = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))]
    
    subject_dir = df[df['set'].str.contains(args.which_set)].index.tolist()
    subject_dir = [os.path.join(args.data_dir, f"{pid}") for pid in subject_dir]
    print(f"data size: {len(subject_dir)}")
        
    hist_dataset = {}

    for idx, current_subject in enumerate(subject_dir):
        start = time.time()

        if not os.path.isdir(current_subject):
            continue
        sub_name = current_subject.split("/")[-1]
        print("Volume Nr: {} Processing Data from {}/{}".format(idx, current_subject, args.MR_name))

        if modal == 'MR':
            MR_data = nib.load(os.path.join(current_subject, args.MR_name))
        else:
            MR_data = nib.load(os.path.join(current_subject, args.CT_name))
        MR_data = np.asanyarray(MR_data.dataobj)
        MR_data = np.nan_to_num(MR_data)

        MR_data = transpose_LPS_to_ITKSNAP_position(MR_data, args.plane)

        # calculate histogram
        histograms, _ = np.histogram(MR_data, bins=num_bins, range=(0.001, 1))
        normalized_histograms = histograms / histograms.sum(keepdims=True)
        normalized_histograms *= scale
        fig_out_path = f'/root_dir/datasets/{modal}_hists_global_{args.height}/{sub_name}.png'
        os.makedirs(os.path.dirname(fig_out_path), exist_ok=True)
        if visualize:
            visualize_histogram(normalized_histograms, num_bins, fig_out_path)
        
        cum_hist = np.cumsum(normalized_histograms)
        fig_out_path = f'/root_dir/datasets/{modal}_hists_global_{args.height}/{sub_name}_cum.png'
        if visualize:
            visualize_histogram(cum_hist, num_bins, fig_out_path)

        hist_diff = np.diff(normalized_histograms)
        hist_diff = np.insert(hist_diff, 0, hist_diff[0])
        hist_diff *= scale
        fig_out_path = f'/root_dir/datasets/{modal}_hists_global_{args.height}/{sub_name}_diff.png'
        if visualize:
            visualize_histogram(hist_diff, num_bins, fig_out_path)
        combined_histogram = np.stack((normalized_histograms, cum_hist, hist_diff), axis=1)
        combined_histogram = np.expand_dims(combined_histogram, axis=-1) # num_bins, 3, 1
        
        hist_dataset[sub_name] = combined_histogram
        print(combined_histogram.shape)
        end = time.time() - start

        print("Volume: {} Finished Data Reading and Appending in {:.3f} seconds.".format(idx, end))

        if args.debugging and idx == 2:
            break

    # # Write the hdf5 file
    with open(args.pkl_name, 'wb') as f:
        pickle.dump(hist_dataset, f)

    end_d = time.time() - start_d
    print("Successfully written {} in {:.3f} seconds.".format(args.pkl_name, end_d))
    print(len(hist_dataset.keys()))


def create_hdf5_dataset_avg(args):
    num_bins = 128
    modal = 'MR'
    start_d = time.time()
    visualize = False
    scale = 10

    subject_dir = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))]
    print(f"data size: {len(subject_dir)}")

    train_dir = os.path.join(os.path.dirname(args.data_dir), "train")
    train_subject_dir = [os.path.join(train_dir, d) for d in os.listdir(train_dir)
                         if os.path.isdir(os.path.join(train_dir, d))]
    print(f"train data size: {len(train_subject_dir)}")

    normalized_list = []
    cum_hist_list = []
    hist_diff_list = []

    for idx, current_subject in enumerate(train_subject_dir):
        start = time.time()

        if not os.path.isdir(current_subject):
            continue

        sub_name = os.path.basename(current_subject)
        print(f"Processing {idx + 1}/{len(train_subject_dir)}: {current_subject}")

        MR_data = nib.load(os.path.join(current_subject, args.MR_name)).get_fdata()
        MR_data = np.nan_to_num(MR_data)

        MR_data = transpose_LPS_to_ITKSNAP_position(MR_data, args.plane)

        histograms, _ = np.histogram(MR_data, bins=num_bins, range=(0.001, 1))
        normalized_histograms = histograms / histograms.sum(keepdims=True)
        normalized_histograms *= scale
        normalized_list.append(normalized_histograms)

        cum_hist = np.cumsum(normalized_histograms)
        cum_hist_list.append(cum_hist)

        hist_diff = np.diff(normalized_histograms)
        hist_diff = np.insert(hist_diff, 0, hist_diff[0])
        hist_diff *= scale
        hist_diff_list.append(hist_diff)

        print(f"{current_subject} processed successfully.")

    normalized_histograms_avg = np.mean(normalized_list, axis=0)
    cum_hist_avg = np.mean(cum_hist_list, axis=0)
    hist_diff_avg = np.mean(hist_diff_list, axis=0)

    hist_dataset = {}
    for idx, current_subject in enumerate(subject_dir):
        combined_histogram = np.stack((normalized_histograms_avg, cum_hist_avg, hist_diff_avg), axis=1)
        combined_histogram = np.expand_dims(combined_histogram, axis=-1)  # num_bins, 3, 1

        sub_name = os.path.basename(current_subject)
        hist_dataset[sub_name] = combined_histogram
        print(f"Processed {sub_name}: {combined_histogram.shape}")

    with open(args.pkl_name, 'wb') as f:
        pickle.dump(hist_dataset, f)

    end_d = time.time() - start_d
    print(f"Successfully written {args.pkl_name} in {end_d:.3f} seconds.")
    print(f"Total subjects processed: {len(hist_dataset)}")

def create_hdf5_dataset_colin(args):
    num_bins = 128
    epsilon = 0.0000001
    modal = 'MR'
    start_d = time.time()
    visualize = False
    scale = 10
    subject_dir = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))]
    
    subject_dir = df[df['set'].str.contains(args.which_set)].index.tolist()
    subject_dir = [os.path.join(args.data_dir, f"{pid}") for pid in subject_dir]
    print(f"data size: {len(subject_dir)}")
    
    hist_dataset = {}

    colin_path = '/root_dir/datasets/colin27/colin27_t1_tal_lin_preprocessed.nii'
    MR_data = nib.load(colin_path)
    MR_data = np.asanyarray(MR_data.dataobj)
    MR_data = np.nan_to_num(MR_data)

    # calculate histogram
    histograms, _ = np.histogram(MR_data, bins=num_bins, range=(0.001, 1))
    normalized_histograms = histograms / histograms.sum(keepdims=True)
    normalized_histograms *= scale
    fig_out_path = f'/root_dir/datasets/colin_hists_global_ver3/hist.png'
    os.makedirs(os.path.dirname(fig_out_path), exist_ok=True)
    if visualize:
        visualize_histogram(normalized_histograms, num_bins, fig_out_path)
    
    cum_hist = np.cumsum(normalized_histograms)
    fig_out_path = f'/root_dir/datasets/colin_hists_global_ver3/hist_cum.png'
    if visualize:
        visualize_histogram(cum_hist, num_bins, fig_out_path)

    hist_diff = np.diff(normalized_histograms)
    hist_diff = np.insert(hist_diff, 0, hist_diff[0])
    hist_diff *= scale
    fig_out_path = f'/root_dir/datasets/colin_hists_global_ver3/hist_diff.png'
    if visualize:
        visualize_histogram(hist_diff, num_bins, fig_out_path)
    combined_histogram = np.stack((normalized_histograms, cum_hist, hist_diff), axis=1)
    combined_histogram = np.expand_dims(combined_histogram, axis=-1) # num_bins, 3, 1
    
    for idx, current_subject in enumerate(subject_dir):
        if not os.path.isdir(current_subject):
            continue
        sub_name = current_subject.split("/")[-1]
        print("Volume Nr: {} Processing Data from {}/{}".format(idx, current_subject, args.MR_name))

        hist_dataset[sub_name] = combined_histogram
        print(combined_histogram.shape)

    # # Write the hdf5 file
    with open(args.pkl_name, 'wb') as f:
        pickle.dump(hist_dataset, f)

    end_d = time.time() - start_d
    print("Successfully written {} in {:.3f} seconds.".format(args.pkl_name, end_d))
    print(len(hist_dataset.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HDF5-Creation')

    parser.add_argument('--pkl_name', type=str, default="testsuite_2.hdf5",
                        help='path and name of hdf5-dataset (default: testsuite_2.hdf5)')
    parser.add_argument('--plane', type=str, default="axial", choices=["axial", "coronal", "sagittal"],
                        help="Which plane to put into file (axial (default), coronal or sagittal)")
    parser.add_argument('--hist_type', type=str)
    parser.add_argument('--which_set', type=str)
    parser.add_argument("--debugging", action='store_true')
    parser.add_argument('--data_csv', type=str)
    parser.add_argument('--pattern', type=str, default="*", help="Pattern to match files in directory.")
    parser.add_argument('--height', type=int, default=128, help='Height of Image (Default 128)')
    parser.add_argument('--width', type=int, default=128, help='Width of Image (Default 128)')
    parser.add_argument('--data_dir', type=str, default="/testsuite", help="Directory with images to load")
    parser.add_argument('--CT_name', type=str)
    parser.add_argument('--MR_name', type=str)

    args = parser.parse_args()

    print(args)
    if args.hist_type == 'avg':
        create_hdf5_dataset_avg(args)
    elif args.hist_type == 'colin':
        create_hdf5_dataset_colin(args)
    else:
        create_hdf5_dataset(args)
