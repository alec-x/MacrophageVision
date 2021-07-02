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
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.macs_frame = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.macs_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_green = os.path.join(self.root_dir,
                                self.macs_frame["green"].iloc[idx])                                
        sample = None
        try:
            #print(img_name_green)
            image_green = io.imread(img_name_green)
            #image_mito = io.imread(img_name_mito)
            #image = np.stack((image_green, image_mito))
            image = np.expand_dims(image_green, axis=0)
            label = self.macs_frame["label"].iloc[idx]
            sample = {'image': image, 'label': label, 'path': img_name_green}
        except Exception as e:
            print(img_name_green)
            print(e)
        if self.transform:
            sample = self.transform(sample)
        return sample