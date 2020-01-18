import torch
import torch.nn as nn
import torch.nn.functional as F


class CDANNSourceGenerator(nn.Module):

    def __init__(self, z_dim, n_classes):
        super(CDANNSourceGenerator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Sequential(
                nn.Linear(z_dim + n_classes, 512),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.5),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(512),
                )
        self.fc3 = nn.Linear(512, 1568)

    def forward(self, x, labels):
        batch_size = x.size(0)
        device = x.get_device()
        one_hot = torch.zeros(batch_size, 10).to(device)
        labels = torch.unsqueeze(labels, 1).to(device)
        one_hot = one_hot.scatter_(1, labels.long(), 1)

        x = torch.cat((x, one_hot), dim=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = torch.tanh(self.fc3(h))
        return h
