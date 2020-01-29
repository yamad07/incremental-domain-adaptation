import torch.nn as nn
import torch.nn.functional as F
from ....layers.gan import fc_layer


class DANNDomainDiscriminator(nn.Module):

    def __init__(self, n_input):
        super(DANNDomainDiscriminator, self).__init__()
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
        self.fc3 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(1024),
                )
        self.fc4 = nn.Linear(1024, 2)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        h = self.fc4(h)
        return F.log_softmax(h, dim=1)
