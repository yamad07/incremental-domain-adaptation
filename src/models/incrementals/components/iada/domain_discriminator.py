import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                )
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = self.fc3(h)
        return F.log_softmax(h, dim=1)
