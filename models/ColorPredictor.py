import time
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from src.data.make_dataset import FaceImagesDataset, train_transforms
from src.models.modules import CNN_Halfing_Block, CNN_Upsampling_Block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FaceImagesDataset = FaceImagesDataset(root_dir='../data/face_images_raw', transform=train_transforms,
                                      mean_chrominance=0)
FaceDataLoader = DataLoader(FaceImagesDataset, shuffle=False, batch_size=1)
colorPredictor = torch.jit.load('ColorizerModel.pt')
colorPredictor.eval()

for x, y in FaceDataLoader:
    # make a prediction
    pred = colorPredictor(x)

    x.to(device)
    y.to(device)

    # get images
    image_pred = torch.squeeze(torch.cat((x, pred), axis=1))
    image_actual = torch.squeeze(torch.cat((x, y), axis=1))

    # rescale images again:
    image_pred[0] = torch.mul(image_pred[0], 100)
    image_pred[1] = torch.mul(image_pred[1], 255)
    image_pred[2] = torch.mul(image_pred[2], 255)

    image_actual[0] = torch.mul(image_actual[0], 100)
    image_actual[1] = torch.mul(image_actual[1], 255)
    image_actual[2] = torch.mul(image_actual[2], 255)

    # swap image axis
    image_pred = image_pred.swapaxes(0, 1)
    image_pred = image_pred.swapaxes(1, 2)

    image_actual = image_actual.swapaxes(0, 1)
    image_actual = image_actual.swapaxes(1, 2)

    image_pred = image_pred.detach().numpy().astype(np.uint8)
    image_actual = image_actual.detach().numpy().astype(np.uint8)

    image = np.concatenate((image_pred, image_actual), axis=1)

    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    image_actual = cv2.cvtColor(image_actual, cv2.COLOR_LAB2BGR)
    image_actual = cv2.cvtColor(image_actual, cv2.COLOR_BGR2GRAY)
    image_actual = np.stack((image_actual,) * 3, axis=-1)

    image = np.concatenate((image_actual, image), axis=1)

    cv2.imshow('', image)
    cv2.waitKey(0)
