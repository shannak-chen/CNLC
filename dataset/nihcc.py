""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
import pandas as pd
import numpy as np

import os
import torch
import logging

from PIL import Image


_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class NIHCC(Dataset):
    def __init__(self, csv, file, transform = None):

        self.to_tensor = transforms.ToTensor()
        self.transform = transform

        self.file_path = file
        self.entry = pd.read_csv(csv, header=0)
        self.data = []
        self.targets = []
        for i in range(len(self.entry)):
            sample = self.entry.iloc[i]
            image_file = sample[0]
            target = sample[1]
            img = Image.open(self.file_path + image_file).convert("RGB")
            self.data.append(img)
            self.targets.append(target)
        self.data = np.array(self.data, dtype=object)

    def __len__(self):
        return len(self.entry)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        return img, target, index
