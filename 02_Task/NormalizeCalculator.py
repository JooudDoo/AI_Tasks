import os
from os.path import join as pjoin

from tqdm import tqdm

import cv2
import numpy

import albumentations as A
import torch
from torch.utils.data import Dataset

def to_tensor(x, **kwargs):
    return torch.tensor(x.transpose(2, 0, 1).astype('float32'))

prepare_to_network = A.Lambda(image=to_tensor, mask=to_tensor)

class RoadsDataset(Dataset):
    def __init__(self, values_dir, labels_dir, class_rgb_values=None, transform=None, readyToNetwork=None):
        self.values_dir = values_dir
        self.labels_dir = labels_dir
        self.class_rgb_values = class_rgb_values
        self.images = [pjoin(self.values_dir, filename) for filename in sorted(os.listdir(self.values_dir))]
        self.labels = [pjoin(self.labels_dir, filename) for filename in sorted(os.listdir(self.labels_dir))]
        self.transform = transform
        self.readyToNetwork = readyToNetwork

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        
        # label = one_hot_encode(label, self.class_rgb_values).astype('float')

        if self.transform:
            sample = self.transform(image=image, mask=label)
            image, label = sample['image'], sample['mask']
        if self.readyToNetwork:
            sample = self.readyToNetwork(image=image, mask=label)
            image, label = sample['image'], sample['mask']
        return image, label


dataset = RoadsDataset("dataset/tiff/train", "dataset/tiff/train_labels", transform=prepare_to_network)

images_means = torch.empty(size=(len(dataset), 3))
images_stds = torch.empty(size=(len(dataset), 3))
labels_means = torch.empty(size=(len(dataset), 3))
labels_stds = torch.empty(size=(len(dataset), 3))

for ind, (image, label) in enumerate(tqdm(dataset)):
    images_means[ind] = (torch.mean(image, dim=[1,2]))
    images_stds[ind] = (torch.std(image, dim=[1,2]))
    labels_means[ind] = (torch.mean(label, dim=[1,2]))
    labels_stds[ind] = (torch.std(label, dim=[1,2]))

images_mean = images_means.mean(dim=[0])
images_std =  images_stds.mean(dim=[0])
labels_mean = labels_means.mean(dim=[0])
labels_std = labels_stds.mean(dim=[0])

print(f"Images means: {images_mean/255}")
print(f"Images std: {images_std/255}")
print(f"Labels means: {labels_mean/255}")
print(f"Images std: {labels_std/255}")