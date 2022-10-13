import os
from skimage import io, transform
import torch
from torch.utils.data import dataset
from PIL import Image
import re
import numpy as np


class MyDataset(dataset.Dataset):
    def __init__(self, root_dir, labels, Transform=None):
        self.root_dir = root_dir
        self.transform = Transform
        self.labels = labels

        self.data = []
        files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        for file in files:
            self.data.append(Image.open(root_dir + file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        label = self.labels[idx]
        sample = {'label': label, 'image': image}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
