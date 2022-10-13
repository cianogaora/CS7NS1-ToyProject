import os
import cv2
import torch
from torch.utils.data import dataset
from PIL import Image



class MyDataset(dataset.Dataset):
    def __init__(self, root_dir, labels, Transform=None):
        self.root_dir = root_dir
        self.transform = Transform
        self.labels = labels

        self.data = []
        files = [f for f in os.listdir(root_dir)]
        for file in files:
            self.data.append(cv2.imread(root_dir + file))
            # with Image.open(root_dir + file) as im:
            #     self.data.append(im)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        sample = {'label': label, 'image': image}

        return sample
