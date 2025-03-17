# Slice-Consistent 3D Volumetric Brain CT-to-MRI Translation with 2D Brownian Bridge Diffusion Model

**Early accepted at MICCAI 2024**

[[Project Page]](https://micv-yonsei.github.io/ct2mri2024/) [[paper]](https://papers.miccai.org/miccai-2024/paper/0531_paper.pdf) [[arXiv]](https://arxiv.org/pdf/2407.05059) 

## Requirements

```
conda env create -f environment.yml
conda activate ct2mri
```

## Dataset Preparation

### Directory Structure :
```
data/
    train/
        patient_1/
            ct.nii.gz
            mr.nii.gz
        patient_2/
            ...
    valid/
        patient_100/
            ct.nii.gz
            mr.nii.gz
        ...
    test/
        patient_200/
            ct.nii.gz
            mr.nii.gz
        ...
```
## Selecting a directory to run scripts:
```commandline
cd yourpathtotheprojectfolder/CT2MRI-CustomDataset
```                                             

### Generate HDF5 Files :
```commandline
sh shell/data/make_hdf5.sh
```
* Outputs HDF5 files to `processed_data/`.
### Generate Histograms for Style Key Conditioning :
```commandline
sh shell/data/make_hist_dataset.sh
```
* Outputs .pkl files to `processed_data/`.

## Training

### Start Training :
```commandline
sh shell/train/train.sh
```

## Testing

### Start Testing :
```commandline
sh shell/test/test.sh
```

## Final Project Structure
```
CT2MRI-CustomDataset/
│
├── brain_dataset_utils/              
│      
├── configs/                                               
│
├── datasets/                          
│
├── model/                           
│
├── runners/                           
│
├── shell/                            
│   ├── data/
│   │   ├── make_hdf5.sh
│   │   ├── make_hist_dataset.sh
│   ├── test/  
│   │   ├── test.sh              
│   ├── train/
│       ├── train.sh
│
├── data/                              
│   ├── train/                        
│   │   ├── patient_1/
│   │   │   ├── CT.nii.gz             
│   │   │   └── MRI.nii.gz             
│   ├── val/                           
│   │   ├── patient_50/
│   │   │   ├── CT.nii.gz
│   │   │   └── MRI.nii.gz
│   └── test/                          
│       ├── patient_200/
│       │   ├── CT.nii.gz
│       │   └── MRI.nii.gz
│       
│
├── processed_data/                    
│   ├── 250_train_axial.hdf5          
│   ├── 250_val_axial.hdf5             
│   ├── 250_test_axial.hdf5            
│   ├── MR_hist_global_250_train_axial_colin.pkl  
│   └── ...
│
├── logs/                             
│   ├── hdf5/
│   │   ├── 250_train_axial.log
│   │   ├── 250_val_axial.log
│   │   └── ...
│   └── hist/
│       ├── MR_hist_global_250_train_axial_colin.log
│       └── ...
│
├── results/                           
│   ├── ct2mr_250/
│   │   ├── experiment_1/
│   │   │   ├── checkpoint/
│   │   │   │   ├── last_model.pth
│   │   │   │   └── last_optim_sche.pth
│   │   │   └── samples/               
│   │   └── ...
│   └── ...
│ 
│ 
other files (main.py, Register.py, utils.py etc.)
```

## Acknowledgement

Our code was implemented based on the code from [BBDM](https://github.com/xuekt98/BBDM). We are grateful to Bo Li, Kai-Tao Xue, et al.


