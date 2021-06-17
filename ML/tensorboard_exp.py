from argparse import ArgumentParser as arg_parser
from MacDataset import MacDataset
import macnet
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision

# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def equal_classes_sampler(df):

    class_count = np.array((df["label"].value_counts()))
    weight = 1. / class_count
    labels = list(df['label'])
    weights = np.array([weight[label] for label in labels])
    samples_weight = torch.from_numpy(weights).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def main(raw_args=None):
    parser = arg_parser(description="Train CNN from dataset")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw images.")
    parser.add_argument("-b", action="store", type=int, \
                        help="batch size when training")
    parser.add_argument("-l", action="store", type=str, \
                        help="log output")   
    parser.set_defaults(b=8, l="runs/default_run")
    
    args = parser.parse_args()


    # Create data loaders.
    csv_path = args.path + '\\' + 'labels.csv' 
    macs_data = MacDataset(root_dir=args.path, csv_file=csv_path)

    sampler = equal_classes_sampler(macs_data.macs_frame)

    
    dataloader = DataLoader(macs_data, batch_size=args.b, sampler=sampler,
                            shuffle=False, num_workers=0)    

    for data in dataloader:
        print("\nShape of X [N, C, H, W]: ", data["image"].shape)
        print("Shape of y: ", data["label"].shape, data["label"].dtype)
        break                            
    
    # run log output
    writer = SummaryWriter(args.l)

    # get some random training images
    dataiter = iter(dataloader)
    data = dataiter.next()["image"]
    # create grid of images
    print(data.shape)
    img_grid = torchvision.utils.make_grid(data.float())

    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)
    
    writer.add_graph(macnet.Net(), data.float())
    writer.close()  

    dataloader = DataLoader(macs_data, batch_size=100, sampler=sampler,
                            shuffle=False, num_workers=0)
                
if __name__=="__main__":
    main()