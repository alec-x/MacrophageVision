from argparse import ArgumentParser as arg_parser
from os import urandom
from MacDataset import MacDataset
import macnet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def equal_classes_sampler(df):

    class_count = np.array((df["label"].value_counts()))
    weight = 1. / class_count
    labels = list(df['label'])
    weights = np.array([weight[label] for label in labels])
    samples_weight = torch.from_numpy(weights).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

class standardize_input(object):
    # single channel
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = (image - self.mean)/self.std
        return {'image': torch.from_numpy(image),
                'label': label}

class rotate_90_input(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        num_rot = urandom.randint(0, 3)
        image = np.rot90(image,num_rot)
        return {'image': torch.from_numpy(image),
                'label': label}    

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    
    # Make input tensor require gradient
    X.requires_grad_()
    
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #forward pass
    scores = model(X.float())
    # print(scores.shape) # torch.Size([5, 1000]) since 5 images, 1000 classes
    scores = (scores.gather(1, y.view(-1, 1)).squeeze())
    # print(scores.shape) # torch.Size([5])

    # print(scores) #tensor([24.1313, 25.1475, 38.8825, 25.4514, 30.2723], grad_fn=<SqueezeBackward0>)
    
    #backward pass
    # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
    # print(scores.shape[0]) # 5
    # print(torch.FloatTensor([1.0]*scores.shape[0])) # tensor([1., 1., 1., 1., 1.])
    scores.backward(torch.FloatTensor([1.0]*scores.shape[0]))
    
    #saliency
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    
    # torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
    
    # Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor 
    # in the given dimension dim. And indices is the index location of each maximum value found (argmax).
    # If keepdim is True, the output tensors are of the same size as input except in the dimension dim 
    # where they are of size 1. Otherwise, dim is squeezed (see torch.squeeze()), 
    # resulting in the output tensors having 1 fewer dimension than input.
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def main(raw_args=None):
    parser = arg_parser(description="Train CNN from dataset")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw images.")
    parser.add_argument("-b", action="store", type=int, \
                        help="batch size when training")
    parser.add_argument("-l", action="store", type=str, \
                        help="log output")   
    parser.add_argument("-n", action="store", type=int, \
                        help="number of folds to do for k-fold validation")                           
    parser.set_defaults(b=1, l=".\\runs\\default_run", n=5)
    
    args = parser.parse_args()


    # Create data loaders.
    csv_path = args.path + '\\' + 'labels.csv' 
    raw_data = pd.read_csv(csv_path)
    
    print("Calculating mean and stdev of dataset for standardization")
    calc_data = MacDataset(root_dir=args.path, dataframe=raw_data)
    #calc_loader = DataLoader(calc_data, batch_size=len(calc_data), num_workers=0)
    #data = next(iter(calc_loader))
    data_mean = 179.49530029296875 # data["image"].float().mean().item()
    data_std = 26.82181739807129 # data["image"].float().std().item()
    
    print("loading data")
    all_transforms = transforms.Compose([standardize_input(data_mean,data_std)])
    
    train_data = MacDataset(root_dir=args.path, dataframe=calc_data.macs_frame,
                                transform=all_transforms)

    train_sampler = equal_classes_sampler(train_data.macs_frame)
    
    dataloader = DataLoader(train_data, batch_size=args.b, sampler=train_sampler,
                            shuffle=False, num_workers=0)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    
    model = macnet.Net()
    model.load_state_dict(torch.load("model"))
    model.eval()

    dataiter = iter(dataloader)
    for i in range(20):
        data = dataiter.next()
        X = data["image"]
        y = data["label"]
        X.requires_grad_()
        scores = model(X.float())
        output_max = scores[0,scores.argmax()]
        output_max.backward()
        saliency,_ = torch.max(X.grad.data.abs(), dim=1) 
        saliency = saliency.reshape(96,96)
        image = X.reshape(96,96)

        #print(saliency.detach().numpy())
        #print(image.detach().numpy())
        # Visualize the image and the saliency map
        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image.detach().numpy())
        ax[0].axis('off')
        ax[1].imshow(saliency.detach().numpy(), cmap='hot')
        ax[1].axis('off')
        plt.tight_layout()
        fig.suptitle('The Image and Its Saliency Map')
        
        plt.show()
        input()
        plt.close()
if __name__=="__main__":
    main()