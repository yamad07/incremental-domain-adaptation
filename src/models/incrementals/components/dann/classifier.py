import torch
import torch.nn as nn
import torch.nn.functional as F


class DANNClassifier(nn.Module):

    def __init__(self):
        super(DANNClassifier, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(1568, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512)
                )
        self.fc2 = nn.Sequential(
                nn.Linear(512, 10),
                )

    def forward(self, feature):
        h = self.fc1(feature)
        return F.log_softmax(self.fc2(h), dim=1)
