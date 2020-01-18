import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ....layers.gan import fc_layer


class DANNSourceGenerator(nn.Module):

    def __init__(self, z_dim, num_features=1568, n_channels=32):
        super(DANNSourceGenerator, self).__init__()
        self.z_dim = z_dim
        self.n_channels = n_channels
        self.fc1 = nn.Sequential(
                nn.Linear(z_dim, 512),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.5),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(512),
                )
        self.fc3 = nn.Linear(512, num_features)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        h = torch.tanh(self.fc3(h))
        b = x.size(0)
        img_size = h.view(b, self.n_channels, -1).size(2)
        img_size = int(np.sqrt(img_size))
        h = h.view(b, self.n_channels, img_size, img_size)
        return h


class DANNSourceDiscriminator(nn.Module):

    def __init__(self, n_input):
        super(DANNSourceDiscriminator, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(n_input, 512),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(512),
                )
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        return F.log_softmax(h, dim=1)
