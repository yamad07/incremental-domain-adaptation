import torch.nn as nn
import torch.nn.functional as F
from ....layers.gan import fc_layer


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                )
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x, use_log_softmax=True):
        h = F.leaky_relu(self.fc1(x))
        h = self.fc4(h)
        return h

    def prob(self, x):
        h = self.forward(x)
        if use_log_softmax:
            return F.log_softmax(h, dim=1)
        else:
            return F.softmax(h, dim=1)
