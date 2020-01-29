import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ....layers.gan import fc_layer


class DANNSourceGenerator(nn.Module):

    def __init__(self, z_dim, num_features=1568, n_channels=64):
        super(DANNSourceGenerator, self).__init__()
        self.z_dim = z_dim
        self.n_channels = n_channels
        self.fc1 = nn.Sequential(
                nn.Linear(z_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                # nn.BatchNorm1d(1024),
                )
        self.fc3 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                # nn.BatchNorm1d(1024),
                )
        self.fc4 = nn.Sequential(
                nn.Linear(1024, num_features),
                )
    def forward(self, x):
        h = self.fc1(x)
        h = self.fc3(h)
        h = F.relu(self.fc4(h))
        b = x.size(0)
        img_size = h.view(b, self.n_channels, -1).size(2)
        img_size = int(np.sqrt(img_size))
        h = h.view(b, self.n_channels, img_size, img_size)
        return h


class DANNSourceDiscriminator(nn.Module):

    def __init__(self, n_input):
        super(DANNSourceDiscriminator, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(n_input, 1024),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(1024),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(1024),
                )
        self.fc4 = nn.Linear(1024, 2)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc4(h)
        return F.log_softmax(h, dim=1)

class DANNConvSourceDiscriminator(nn.Module):

    def __init__(self, n_input):
        super(DANNConvSourceDiscriminator, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                )
        self.fc2 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                )
        self.fc4 = nn.Linear(288, 2)

    def forward(self, x):
        size = x.size()
        channel_size = 64
        x = x.view(size[0], channel_size, -1)
        wh = x.size(2)
        x = x.view(size[0], channel_size, int(math.sqrt(wh)), int(math.sqrt(wh)))
        h = self.fc1(x)
        h = self.fc2(h)
        b = h.size(0)
        h = h.view(b, -1)
        h = self.fc4(h)
        return F.log_softmax(h, dim=1)
