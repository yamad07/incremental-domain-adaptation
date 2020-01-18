import torch.nn as nn
import torch.nn.functional as F
from ....layers.gan import fc_layer


class DANNDomainDiscriminator(nn.Module):

    def __init__(self, n_input):
        super(DANNDomainDiscriminator, self).__init__()
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
