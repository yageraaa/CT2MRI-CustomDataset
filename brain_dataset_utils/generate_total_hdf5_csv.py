import argparse
import time
import h5py
import os
import numpy as np
import nibabel as nib
import pandas as pd

def make_transpose_dict(axcodes):
    dim_match = {} # Nifti uses RAH+ #
    for di, code in enumerate(axcodes):
        if code in ['R', 'L']:
            dim_match[0] = di
        elif code in ['A', 'P']:
            dim_match[1] = di
        elif code in ['S', 'I']:
            dim_match[2] = di
    transpose_axes = tuple([dim_match[0], dim_match[1], dim_match[2]])

    return dim_match, transpose_axes

def transform_axial(vol, coronal2axial=True):
    """
    Function to transform volume into Axial axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2axial: transform from coronal to axial = True (default),
                               transform from axial to coronal = False
    :return: np.ndarray: transformed image volume
    """
    if coronal2axial:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])


def transform_sagittal(vol, coronal2sagittal=True):
    """
    Function to transform volume into Sagittal axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2sagittal: transform from coronal to sagittal = True (default),
                                transform from sagittal to coronal = False
    :return: np.ndarray: transformed image volume
    """
    if coronal2sagittal:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])


def filter_blank_slices_thick(CT_data, MR_data, threshold=50):
    # Get indices of all slices with more than threshold labels/pixels
    binaray_label = np.where(CT_data > 1e-15, 1, 0)
    select_slices = (np.sum(binaray_label, axis=(0, 1)) > threshold)

    # Если нет подходящих срезов, вернуть исходные данные с предупреждением
    if np.sum(select_slices) == 0:
        print("WARNING: All slices filtered out! Returning original data.")
        return CT_data, MR_data

    # Retain only slices with more than threshold labels/pixels
    CT_data = CT_data[:, :, select_slices]
    MR_data = MR_data[:, :, select_slices]

    true_indices = np.where(select_slices)[0]
    first_true_index = true_indices[0] if true_indices.size > 0 else None
    last_true_index = true_indices[-1] if true_indices.size > 0 else None
    print(f"Retained slices: {first_true_index} ~ {last_true_index} ({len(true_indices)} slices)")
    return CT_data, MR_data


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


def create_hdf5_dataset(args):
    start_d = time.time()

    subject_dir = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))]
    print(f"Found {len(subject_dir)} subjects in {args.data_dir}.")

    subject_dir = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))]
    print(f"data size: {len(subject_dir)}")

    CT_dataset = np.ndarray(shape=(args.height, args.width, 0), dtype=np.float32)
    MR_dataset = np.ndarray(shape=(args.height, args.width, 0), dtype=np.float32)
    index_dataset = np.ndarray(shape=(0), dtype=np.uint8)
    max_index_dataset = np.ndarray(shape=(0), dtype=np.uint8)
    subjects = []

    for idx, current_subject in enumerate(subject_dir):
        start = time.time()

        if not os.path.isdir(current_subject):
            continue

        print("Volume Nr: {} Processing Data from {}/{}".format(idx, current_subject, args.CT_name))

        CT_data = nib.load(os.path.join(current_subject, args.CT_name))
        CT_data = np.asanyarray(CT_data.dataobj)
        CT_data = np.nan_to_num(CT_data)
        MR_data = nib.load(os.path.join(current_subject, args.MR_name))
        MR_data = np.asanyarray(MR_data.dataobj)
        MR_data = np.nan_to_num(MR_data)

        print("CT data shape:", CT_data.shape)  # Отладочный вывод
        print("MR data shape:", MR_data.shape)  # Отладочный вывод

        CT_data = transpose_LPS_to_ITKSNAP_position(CT_data, args.plane)
        MR_data = transpose_LPS_to_ITKSNAP_position(MR_data, args.plane)
        print(f"After transpose - CT shape: {CT_data.shape}, MR shape: {MR_data.shape}")

        CT_data, MR_data = filter_blank_slices_thick(CT_data, MR_data, threshold=50)

        # # Append finally processed images to arrays
        CT_dataset = np.append(CT_dataset, CT_data, axis=2)
        MR_dataset = np.append(MR_dataset, MR_data, axis=2)
        index_dataset = np.append(index_dataset, np.arange(CT_data.shape[2], dtype=np.uint8), axis=0)
        max_index_dataset = np.append(max_index_dataset,
                                      np.full(CT_data.shape[2], CT_data.shape[2] - 1, dtype=np.uint8), axis=0)
        sub_name = current_subject.split("/")[-1]
        subjects.extend([sub_name.encode("ascii", "ignore")] * CT_data.shape[2])
        # subjects.append(sub_name.encode("ascii", "ignore"))

        end = time.time() - start

        print("Volume: {} Finished Data Reading and Appending in {:.3f} seconds.".format(idx, end))

        if args.debugging and idx == 2:
            break

    index_dataset = np.transpose(np.vstack((index_dataset, max_index_dataset))).astype(np.uint8)

    # # Write the hdf5 file
    with h5py.File(args.hdf5_name, "w") as hf:
        hf.create_dataset('CT_dataset', data=CT_dataset, compression='gzip')
        hf.create_dataset('MR_dataset', data=MR_dataset, compression='gzip')
        hf.create_dataset('index_dataset', data=index_dataset, compression='gzip')

        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset("subject", data=subjects, dtype=dt, compression="gzip")

    end_d = time.time() - start_d
    print("Successfully written {} in {:.3f} seconds.".format(args.hdf5_name, end_d))
    print(index_dataset.shape)
    print(CT_dataset.shape)
    print(len(subjects))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HDF5-Creation')
    parser.add_argument('--hdf5_name', type=str, default="processed_data/250_train_axial.hdf5",
                        help='Path and name of HDF5 file (default: processed_data/250_train_axial.hdf5)')
    parser.add_argument('--plane', type=str, default="axial", choices=["axial", "coronal", "sagittal"],
                        help="Plane to process (default: axial)")
    parser.add_argument('--which_set', type=str, required=True, help="Dataset type (train, val, test)")
    parser.add_argument('--height', type=int, default=250, help='Height of image (default: 250)')
    parser.add_argument('--width', type=int, default=250, help='Width of image (default: 250)')
    parser.add_argument('--data_dir', type=str, required=True, help="Directory with images to load")
    parser.add_argument('--CT_name', type=str, default="ct.nii.gz", help="Name of CT file (default: ct.nii.gz)")
    parser.add_argument('--MR_name', type=str, default="mr.nii.gz", help="Name of MRI file (default: mr.nii.gz)")
    parser.add_argument("--debugging", action='store_true', help="Enable debugging mode.")

    args = parser.parse_args()
    create_hdf5_dataset(args)
