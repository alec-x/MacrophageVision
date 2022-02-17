import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid

class Net(nn.Module):

    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3)
        self.conv1a = nn.Conv2d(32, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2a = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3a = nn.Conv2d(128, 128, 3)

        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # TODO: NORM INPUT call at top of forward
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv1a(x)), (2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv2a(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv3a(x)), (2, 2))   
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)
        #return sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features