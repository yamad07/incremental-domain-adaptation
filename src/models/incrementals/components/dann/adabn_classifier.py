import torch
import torch.nn as nn
import torch.nn.functional as F
from ..adabn import AdaptiveBatchNorm1d


class AdaBNClassifier(nn.Module):

    def __init__(self, n_input):
        super(AdaBNClassifier, self).__init__()
        self.adabn = AdaptiveBatchNorm1d(n_input)
        self.fc1 = nn.Sequential(
                nn.Linear(n_input, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256)
                )
        self.fc2 = nn.Sequential(
                nn.Linear(256, 10),
                )

    def forward(self, feature):
        feature = self.adabn(feature)
        h = self.fc1(feature)
        return F.log_softmax(self.fc2(h), dim=1)
