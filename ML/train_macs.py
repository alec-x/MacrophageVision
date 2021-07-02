from argparse import ArgumentParser as arg_parser
import random
from macdataset import MacDataset
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
import torch.nn.functional as F
from torchvision.transforms import functional

def equal_classes_sampler(df):

    class_count = np.array((df["label"].value_counts()))
    weight = 1. / class_count
    labels = list(df['label'])
    weights = np.array([weight[label] for label in labels])
    samples_weight = torch.from_numpy(weights).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler
def visualize_samples(dataloader):
    dataiter = iter(dataloader)
    data = dataiter.next()
    X = data["image"][0][0]
    plt.imshow(X)
    plt.show()
    input()

class standardize_input(object):
    # single channel
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        mean = np.mean(image)
        stdev = np.std(image)
        image = (image - mean)/stdev
        return {'image': torch.from_numpy(image),
                'label': label}

class rotate_90_input(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        num_rot = random.randint(0, 3)
        image = torch.rot90(image,num_rot, [1,2])
        return {'image': image,
                'label': label}    

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
    parser.set_defaults(b=4, l=".\\runs\\default_run", n=5)
    
    args = parser.parse_args()


    # Create data loaders.
    csv_path = args.path + '\\' + 'labels.csv' 
    raw_data = pd.read_csv(csv_path)
    split_data = np.array_split(raw_data.sample(frac=1), args.n)
    
    print("Calculating mean and stdev of dataset for standardization")
    #calc_data = MacDataset(root_dir=args.path, dataframe=raw_data)
    #calc_loader = DataLoader(calc_data, batch_size=len(calc_data), num_workers=0)
    #data = next(iter(calc_loader))
    #data_mean = data["image"].float().mean().item()
    #data_std = data["image"].float().std().item()
    #data_mean = 179.49530029296875 
    #data_std = 26.82181739807129 
    testing_errors = []
    training_errors = []
    for i in range(len(split_data)):
        print("\n FOLD " + str(i + 1) + " OF " + str(args.n))
        print("=========================================================\n")

        train_idx = list(range(args.n))
        train_idx.remove(i)
        train_idx_start = train_idx.pop()
        train_df = split_data[train_idx_start].copy()
        for idx in train_idx:
            train_df = train_df.append(split_data[idx])
        
        train_transforms = transforms.Compose([
            standardize_input(),
            rotate_90_input()
            ])
        test_transforms = transforms.Compose([
            standardize_input()
            ])

        train_data = MacDataset(root_dir=args.path, dataframe=train_df,
                                    transform=train_transforms)
        test_data = MacDataset(root_dir=args.path, dataframe=split_data[i],
                                    transform=test_transforms)

        train_sampler = equal_classes_sampler(train_data.macs_frame)
        test_sampler = equal_classes_sampler(test_data.macs_frame)
        
        dataloader = DataLoader(train_data, batch_size=args.b, sampler=train_sampler,
                                shuffle=False, num_workers=0)

        dataloader_test = DataLoader(test_data, batch_size=args.b, sampler=test_sampler,
                                shuffle=False, num_workers=0)              

        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
        
        model = macnet.Net().to(device)
        #print("\nConvolutional Neural Net Model:")
        #print(model)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print("\nTraining Start")
        
        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            total_done = 0
            correct = 0
            bad_batches = 0
            final_training_acc = 0
            
            for batch, data in enumerate(dataloader):
                try:
                    X, y = data["image"].to(device), data["label"].to(device)

                    # Compute prediction error
                    pred = model(X.float())
                    loss = loss_fn(torch.squeeze(pred), y.float())

                    # Backpropagation
                    model.train()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch % 25 == 0:
                        loss, current = loss.item(), batch * len(X)
                        #print(torch.squeeze(pred).round(), y)
                        #input()
                        correct += (torch.squeeze(pred).round() == y).type(torch.float).sum().item()
                        total_done += args.b
                        training_acc = correct/total_done
                        final_training_acc = training_acc
                        print(f"Avg. Loss: {loss:>7f}, Accuracy: {training_acc:>.2%} [{current:>5d}/{size:>5d}]", end="\r")
                except:
                    bad_batches += 1
            print()
            print("bad batches:" + str(bad_batches))
            return final_training_acc
            

        def test(dataloader, model):
            size = len(dataloader.dataset)
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for data in dataloader:
                    X, y = data["image"].to(device), data["label"].to(device)
                    pred = model(X.float())
                    test_loss += loss_fn(pred, torch.unsqueeze(y, 1).float()).item()
                    correct += (torch.squeeze(pred).round() == y).type(torch.float).sum().item()
            test_loss /= size
            correct /= size
            print(f"\nTest Error: \nAvg. Loss: {test_loss:>7f}, Accuracy: {correct:>0.2%}\n")
            return correct

        epochs = 5
        training_error = []
        testing_error = []
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            print("\nTraining Error:")
            training_error.append(train(dataloader, model, loss_fn, optimizer))
            testing_error.append(test(dataloader_test, model))
        training_errors.append(statistics.mean(training_error))
        testing_errors.append(statistics.mean(testing_error))

        if i == 0:
            torch.save(model, "./model")

    training_errors = [round(error, 4) for error in training_errors]
    testing_errors = [round(error, 4) for error in testing_errors]
    print("training errors per fold")
    print(training_errors)
    print("testing errors per fold")
    print(testing_errors)

if __name__=="__main__":
    main()