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

        img_name = os.path.join(self.root_dir,
                                self.macs_frame.iloc[idx, 0])
        image = io.imread(img_name)
        macs = self.macs_frame.iloc[idx, 1:]
        macs = np.array([macs])
        macs = macs.astype('float').reshape(-1, 2)
        sample = {'image': image, 'macs': macs}

        if self.transform:
            sample = self.transform(sample)

        return sample