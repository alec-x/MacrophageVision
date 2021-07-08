from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class MacDataset(Dataset):
    """Macrophage dataset."""
    # Derived from https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
    def __init__(self, images, labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]
    def __order__(self):
        return self.order
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            image = self.images[idx]
            label = self.labels[idx]
            sample = [image, label]
        except Exception as e:
            print("sample: " + str(idx) + " did not load")
            print(e)
        if self.transform:
            sample = self.transform(sample)
        return sample