import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
from torch.utils.data import WeightedRandomSampler
import cv2

class standardize_input(object):
    # single channel
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        chans = range(image.shape[0])
        means = [np.mean(image[chan]) for chan in chans]
        stdevs = [np.std(image[chan]) for chan in chans]
        output = torch.Tensor(image)
        for chan in chans:
            output[chan] = (output[chan] - means[chan]) / stdevs[chan]

        return [output, label]
                
class rotate_90_input(object):
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        num_rot = random.randint(0, 3)
        image = torch.rot90(image, num_rot, [1,2])
        return [image, label]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class gaussian_blur(object):
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        ksize = (self.kernel_size, self.kernel_size)
        chans = range(image.shape[0])
        output = torch.Tensor(image)
        for chan in chans:
            output[chan] = torch.Tensor(cv2.GaussianBlur(image[chan].numpy(), ksize, self.sigma))
        return [image, label]

def visualize_samples(dataloader, num_samples):
    dataiter = iter(dataloader)
    data = dataiter.next()
    for _ in range(num_samples):
        X = data["image"][0][0]
        plt.imshow(X)
        plt.show()
        input()

def equal_classes_sampler(labels):
    _, class_count = np.unique(labels, return_counts=True)
    weight = 1. / class_count
    labels = list(labels)
    weights = np.array([weight[int(label)] for label in labels])
    samples_weight = torch.from_numpy(weights).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler
