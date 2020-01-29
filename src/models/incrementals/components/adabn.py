import torch
import torch.nn as nn


class AdaptiveBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)
