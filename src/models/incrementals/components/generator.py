import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...layers.gan import fc_layer


class VGGSourceGenerator(nn.Module):

    def __init__(self, z_dim, num_features=1568, n_channels=32):
        super(VGGSourceGenerator, self).__init__()
        self.z_dim = z_dim
        self.n_channels = n_channels
        self.fc1 = nn.Sequential(
                nn.Linear(z_dim, 512),
                nn.ReLU(512),
                nn.BatchNorm1d(512),
                )
        self.fc5 = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                )
        self.fc6 = nn.Sequential(
                nn.Linear(512, num_features),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc5(h)
        h = self.fc6(h)
        b = x.size(0)
        img_size = h.view(b, self.n_channels, -1).size(2)
        img_size = int(np.sqrt(img_size))
        h = h.view(b, self.n_channels, img_size, img_size)
        return h
