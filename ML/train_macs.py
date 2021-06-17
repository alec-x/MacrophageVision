from argparse import ArgumentParser as arg_parser
from MacDataset import MacDataset
import macnet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision

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
    parser.add_argument("-n", action="store", type=int, \
                        help="number of folds to do for k-fold validation")                           
    parser.set_defaults(b=4, l=".\\runs\\default_run", n=5)
    
    args = parser.parse_args()


    # Create data loaders.
    csv_path = args.path + '\\' + 'labels.csv' 
    raw_data = pd.read_csv(csv_path)
    split_data = np.array_split(raw_data.sample(frac=1), args.n)

    for i in range(len(split_data)):
        print("\n FOLD " + str(i) + " OF " + str(args.n))
        print("=========================================================\n")
        
        train_idx = list(range(args.n))
        train_idx.remove(i)
        train_idx_start = train_idx.pop()
        train_df = split_data[train_idx_start].copy()
        for idx in train_idx:
            train_df = train_df.append(split_data[idx])

        train_data = MacDataset(root_dir=args.path, dataframe=train_df)
        test_data = MacDataset(root_dir=args.path, dataframe=split_data[i])
        train_sampler = equal_classes_sampler(train_data.macs_frame)
        test_sampler = equal_classes_sampler(test_data.macs_frame)
        
        dataloader = DataLoader(train_data, batch_size=args.b, sampler=train_sampler,
                                shuffle=False, num_workers=0)    
        dataloader_test = DataLoader(test_data, batch_size=args.b, sampler=test_sampler,
                                shuffle=False, num_workers=0)
        for data in dataloader:
            print("\nShape of X [N, C, H, W]: ", data["image"].shape)
            print("Shape of y: ", data["label"].shape, data["label"].dtype)
            break                            
        
        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\nUsing {} device".format(device))
        
        model = macnet.Net().to(device)
        print("\nConvolutional Neural Net Model:")
        print(model)
        
        # run log output
        writer = SummaryWriter(args.l)

        # get some random training images
        dataiter = iter(dataloader)
        data = dataiter.next()["image"]

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print("\nTraining Start")
        
        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            total_done = 0
            correct = 0
            bad_batches = 0
            for batch, data in enumerate(dataloader):
                try:
                    X, y = data["image"].to(device), data["label"].to(device)

                    # Compute prediction error
                    pred = model(X.float())
                    loss = loss_fn(torch.squeeze(pred), y.float())

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch % 25 == 0:
                        loss, current = loss.item(), batch * len(X)
                        
                        correct += (torch.squeeze(pred).round() == y).type(torch.float).sum().item()
                        total_done += args.b
                        training_acc = correct/total_done
                        print(f"Avg. Loss: {loss:>7f}, Accuracy: {training_acc:>.2%} [{current:>5d}/{size:>5d}]", end="\r")
                except:
                    bad_batches += 1
            print("bad batches:" + str(bad_batches))
            print()

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

        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            print("\nTraining Error:")
            train(dataloader, model, loss_fn, optimizer)
            test(dataloader_test, model)
    
if __name__=="__main__":
    main()