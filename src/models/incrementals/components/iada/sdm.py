import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceGenerator(nn.Module):

    def __init__(self, z_dim):
        super(SourceGenerator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Sequential(
                nn.Linear(z_dim, 1024),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.5),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(512),
                )
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        h = torch.tanh(self.fc3(h))
        return h


class SourceDiscriminator(nn.Module):

    def __init__(self):
        super(SourceDiscriminator, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(256, 1024),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(1024),
                )
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        return F.log_softmax(h, dim=1)
