import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional
from torch.utils.data import WeightedRandomSampler

class standardize_input(object):
    # single channel
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        mean = np.mean(image)
        stdev = np.std(image)
        image = (image - mean)/stdev
        return [torch.from_numpy(image), label]
                
class rotate_90_input(object):
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        num_rot = random.randint(0, 3)
        image = torch.rot90(image,num_rot, [1,2])
        return [image, label]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
        
class center_crop(object):
    def __init__(self, size_range):
        self.range = size_range
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        orig = image.shape[2]
        crop_size = random.randint(int(self.range[0]/2), int(self.range[1]/2))*2
        p_size = int((orig - crop_size) / 2)
        image = functional.center_crop(image, crop_size)
        image = F.pad(input=image, pad=(p_size, p_size, p_size, p_size), mode='constant', value=0)
        return {'image': image,
                'label': label}

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
