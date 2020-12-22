from file_and_folder_operations import *
import SimpleITK as sitk

# data = load_pickle(r'D:\preprocessed_COVID19\crop_foreground\dataset_properties.pkl')
# print(data)
#
# resample_img,resample_seg = np.load(r'D:\preprocessed_COVID19\resample_normalization\volume-covid19-A-0003.npy')
# resample_img = sitk.GetImageFromArray(resample_img)
# resample_img.SetSpacing(np.array([2.2      , 0.8287265, 0.8287265])[::-1])
#
# resample_seg = sitk.GetImageFromArray(resample_seg)
# resample_seg.SetSpacing(np.array([2.2      , 0.8287265, 0.8287265])[::-1])
# sitk.WriteImage(resample_img, './0003.nii.gz')
# sitk.WriteImage(resample_seg, './0003_seg.nii.gz')
print(round(1.34))
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset


class DealDataset(Dataset):
    def __init__(self):
        xy = np.random.randn(4, 3)
        print(xy)
        print("\n")
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dealDataset = DealDataset()

train_loader2 = DataLoader(dataset=dealDataset,
                           batch_size=1,
                           shuffle=True)
for j in range(5):
    for i, data in enumerate(train_loader2):
        inputs, labels = data

        # inputs, labels = Variable(inputs), Variable(labels)
        print(inputs)
        print(labels)
    print("\n")
    # print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())