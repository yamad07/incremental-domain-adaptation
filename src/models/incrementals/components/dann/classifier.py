import torch
import torch.nn as nn
import torch.nn.functional as F


class DANNClassifier(nn.Module):

    def __init__(self, n_input):
        super(DANNClassifier, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(n_input, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                # nn.BatchNorm1d(1024)
                )
        self.fc2 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                # nn.BatchNorm1d(1024)
                )
        self.fc3 = nn.Sequential(
                nn.Linear(1024, 10),
                )

    def forward(self, feature):
        b = feature.size(0)
        h = self.fc1(feature.view(b, -1))
        h = self.fc2(h)
        return F.log_softmax(self.fc3(h), dim=1)
