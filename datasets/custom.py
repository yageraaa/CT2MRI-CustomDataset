import torch
from torch.utils.data import Dataset

from Register import Registers
from datasets.base import multi_ch_nifti_default_Dataset
import os
import numpy as np
import h5py
import pickle

@Registers.datasets.register_with_name('BraTS_t2f_t1n_aligned_global_hist_context')
class hist_context_BraTS_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"BraTS_t2f_to_t1n_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('target_dataset'))
            B_dataset = np.array(hf.get('source_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))

        hist_type = dataset_config.hist_type
        hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_t1n_{stage}_{dataset_config.plane}_BraTS_.pkl")
        if stage == 'test' and hist_type is not None:
            hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_t1n_{stage}_{dataset_config.plane}_BraTS_"+hist_type+".pkl")   
        print(hist_path)
        with open(hist_path, 'rb') as f:
            self.hist_dict = pickle.load(f)

        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        
    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        out_ori = self.imgs_ori[i] # (3, 160, 160)
        out_cond = self.imgs_cond[i] # (3, 160, 160)
        out_hist = self.hist_dict[out_cond[1].decode('utf-8')]
        out_hist = torch.from_numpy(out_hist).float() # (32, 128, 1)

        return out_ori, out_cond, out_hist
    
@Registers.datasets.register_with_name('BraTS_t2f_t1n_aligned')
class BraTS_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"BraTS_t2f_to_t1n_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('target_dataset'))
            B_dataset = np.array(hf.get('source_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))

        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]


@Registers.datasets.register_with_name('ct2mr_aligned_global_hist_context')
class hist_context_CT2MR_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        self.stage = stage
        hdf5_path = os.path.join(dataset_config.dataset_path,
                                 f"{dataset_config.image_size}_{stage}_{dataset_config.plane}.hdf5")
        print(f"Loading imaging data from: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as hf:
            self.A_dataset = np.array(hf.get('MR_dataset'))
            self.B_dataset = np.array(hf.get('CT_dataset'))
            self.index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            self.subjects = np.array(hf.get("subject"))

        hist_type = dataset_config.hist_type if hasattr(dataset_config, 'hist_type') else None
        hist_basename = f"MR_hist_global_{dataset_config.image_size}_{stage}_{dataset_config.plane}_"
        hist_path = os.path.join(dataset_config.dataset_path,
                                 hist_basename + ("avg.pkl" if stage == 'test' and hist_type else ".pkl"))
        print(f"Loading histograms from: {hist_path}")

        with open(hist_path, 'rb') as f:
            self.hist_dict = pickle.load(f)

        self._validate_patient_ids()
        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(
            self.A_dataset, self.index_dataset, self.subjects,
            self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal
        )
        self.imgs_cond = multi_ch_nifti_default_Dataset(
            self.B_dataset, self.index_dataset, self.subjects,
            self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal
        )

        print(f"Initialized {self.__class__.__name__} with {len(self)} samples")

    def _validate_patient_ids(self):
        missing_ids = []
        valid_ids = []

        for subj in self.subjects:
            pid = subj.decode('utf-8')
            if pid not in self.hist_dict:
                missing_ids.append(pid)
            else:
                valid_ids.append(pid)

        if missing_ids:
            print(f"Warning: Missing histograms for {len(missing_ids)}/{len(self.subjects)} patients")
            print(f"First 5 missing IDs: {missing_ids[:5]}")

            sample_hist = next(iter(self.hist_dict.values())) if self.hist_dict else np.zeros((32, 128, 1))
            default_hist = np.zeros_like(sample_hist)

            for pid in missing_ids:
                self.hist_dict[pid] = default_hist
                print(f"Created default histogram for patient: {pid}")

        if valid_ids:
            sample_hist = self.hist_dict[valid_ids[0]]
            print(f"Histogram shape: {sample_hist.shape}, dtype: {sample_hist.dtype}")
            print(f"First valid patient ID: {valid_ids[0]}")

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, idx):
        try:
            out_ori = self.imgs_ori[idx]
            out_cond = self.imgs_cond[idx]

            patient_id = out_cond[1].decode('utf-8') if isinstance(out_cond[1], bytes) else str(out_cond[1])

            out_hist = self.hist_dict.get(patient_id)
            if out_hist is None:
                print(f"Histogram missing for patient {patient_id}, using zeros")
                out_hist = np.zeros((32, 128, 1), dtype=np.float32)

            return (
                out_ori,
                out_cond,
                torch.from_numpy(out_hist).float()
            )

        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return (
                torch.zeros((3, *self.image_size)),
                torch.zeros((3, *self.image_size)),
                torch.zeros((32, 128, 1))
            )


@Registers.datasets.register_with_name('ct2mr_aligned')
class CT2MR_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"{dataset_config.image_size}_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('MR_dataset'))
            B_dataset = np.array(hf.get('CT_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))
            
        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]

