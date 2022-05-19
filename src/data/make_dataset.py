import glob
import os
import torch
import numpy as np
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import warnings

warnings.filterwarnings("ignore")

DATA_AUGMENTATION_SCALE = 10


train_transforms = A.Compose(
    [
        A.HorizontalFlip(p = 0.5),
        #A.RGBShift(),
        A.RandomResizedCrop(height=128, width=128),
        ToTensorV2()
    ]
)


class FaceImagesDataset(Dataset):

    def __init__(self, root_dir, transform=None, mean_chrominance = False):
        self.root_dir = root_dir
        self.image_paths = os.listdir(root_dir)
        self.transform = transform
        self.mean_chrominance = mean_chrominance

    def __len__(self):
        return DATA_AUGMENTATION_SCALE * len(self.image_paths) # make 10x more data using augmentation

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_filepath = os.path.join(self.root_dir, self.image_paths[idx % (int(self.__len__() / DATA_AUGMENTATION_SCALE))])
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        if (self.transform):
            image = self.transform(image=image)
            image = image['image']
            #swap axes to view data, leave unswapped to feed into CNN
            #image = image.swapaxes(0, 1)
            #image = image.swapaxes(1, 2)
            image = image.float()
            image[0] = torch.div(image[0],100)
            image[1] = torch.div(image[1],255)
            image[2] = torch.div(image[2],255)

        if(self.mean_chrominance):

            return image[0][None,:,:], torch.tensor([[[torch.mean(torch.Tensor.float(image[1]))]], [[torch.mean(torch.Tensor.float(image[2]))]]])
        else:
            return image[0][None,:,:], image[1:]

