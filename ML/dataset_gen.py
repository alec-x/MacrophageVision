from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class MacDataset(Dataset):
    """Macrophage dataset."""
    # Derived from https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.macs_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.macs_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_bf = os.path.join(self.root_dir,
                                self.macs_frame.iloc[idx, 0])
        img_name_mito = os.path.join(self.root_dir,
                                self.macs_frame.iloc[idx, 1])                                
        
        image_bf = io.imread(img_name_bf)
        image_mito = io.imread(img_name_mito)
        image = np.stack((image_bf, image_mito))
        label = self.macs_frame.iloc[idx, 2]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
root = 'C:\\Users\\Alec\\Documents\\Source\\Repos\\MDLMacVis2\\data\\processed\\labeled'
Macs_data = MacDataset(root_dir=root, csv_file=root + '\\' + 'labels.csv')

fig = plt.figure()

for i in range(len(Macs_data)):
    sample = Macs_data[i]

    print(i, sample['image'].shape, sample['label'])
    if i == 3:
        plt.imshow(sample['image'][1])
        plt.show()
        input()
        break