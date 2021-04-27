from argparse import ArgumentParser as arg_parser
from MacDataset import MacDataset
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import macnet
import matplotlib.pyplot as plt
import numpy as np
def equal_classes_sampler(df):

    class_count = np.array((df["label"].value_counts()))
    weight = 1. / class_count
    labels = list(df['label'])
    weights = np.array([weight[label] for label in labels])
    samples_weight = torch.from_numpy(weights).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def main(raw_args=None):
    parser = arg_parser(description="Train CNN from dataset")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw images.")
    parser.add_argument("-b", action="store", type=int, \
                        help="batch size when training")
    
    parser.set_defaults(b=4)
    
    args = parser.parse_args()


    # Create data loaders.
    csv_path = args.path + '\\' + 'labels.csv' 
    macs_data = MacDataset(root_dir=args.path, csv_file=csv_path)

    sampler = equal_classes_sampler(macs_data.macs_frame)

    
    dataloader = DataLoader(macs_data, batch_size=args.b, sampler=sampler,
                            shuffle=False, num_workers=0)    
    dataloader_test = DataLoader(macs_data, batch_size=args.b, sampler=sampler,
                            shuffle=False, num_workers=0)    
    for data in dataloader:
        print("\nShape of X [N, C, H, W]: ", data["image"].shape)
        print("Shape of y: ", data["label"].shape, data["label"].dtype)
        break                            
    
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print("\nUsing {} device".format(device))
    
    model = macnet.Net().to(device)
    print("\nConvolutional Neural Net Model:")
    print(model)
    
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    print("\nTraining Start")
    
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, data in enumerate(dataloader):
            X, y = data["image"].to(device), data["label"].to(device)

            # Compute prediction error
            pred = model(X.float())
            loss = loss_fn(pred, torch.unsqueeze(y, 1).float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 25 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="\r")
    
    def test(dataloader, model):
        size = len(dataloader.dataset)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in dataloader:
                X, y = data["image"].to(device), data["label"].to(device)
                pred = model(X.float())
                test_loss += loss_fn(pred, torch.unsqueeze(y, 1).float()).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer)
        test(dataloader_test, model)
    
if __name__=="__main__":
    main()