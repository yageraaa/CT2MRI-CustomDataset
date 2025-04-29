from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision import transforms
import torch
import torchvision

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision import transforms

class multi_ch_nifti_default_Dataset(Dataset):
    def __init__(self, image_dataset, index_dataset, subjects, radius, image_size=(250, 250), flip=False,
                 to_normal=False):
        self.image_size = image_size
        self.images = image_dataset
        self.indice = index_dataset
        self.subjects = subjects
        self.radius = radius
        self.flip = flip
        self.to_normal = to_normal
        self._length = self.images.shape[2]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        slice_number = self.indice[index, 0]
        max_slice_number = self.indice[index, 1]

        if slice_number < self.radius:
            image = self.images[:, :, index - slice_number:index + self.radius + 1]
            image = np.pad(image, ((0, 0), (0, 0), (self.radius - slice_number, 0)), mode='constant')
        elif slice_number > max_slice_number - self.radius:
            image = self.images[:, :, index - self.radius:index + max_slice_number - slice_number + 1]
            image = np.pad(image, ((0, 0), (0, 0), (0, self.radius + slice_number - max_slice_number)), mode='constant')
        else:
            image = self.images[:, :, index - self.radius:index + self.radius + 1]

        transform = A.Compose([
            A.HorizontalFlip(p=p),
            ToTensorV2(),
        ])

        image = transform(image=image)['image'].float()

        # if self.to_normal:
        #     mean = torch.mean(image)
        #     std = torch.std(image)
        #     if std == 0:
        #         std = 1.0
        #     image = (image - mean) / std


        return image, self.subjects[index]

    def get_subject_names(self):
        return self.subjects