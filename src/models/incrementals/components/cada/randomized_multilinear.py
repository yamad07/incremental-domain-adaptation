import torch
import torch.nn as nn


class RandomizedMultilinear(nn.Module):

    def __init__(self, n_input, n_random_features):
        super(RandomizedMultilinear, self).__init__()
        self.f = nn.Linear(n_input, n_random_features, bias=False)
        torch.nn.init.normal_(self.f.weight)

    def forward(self, input):
        return self.f(input)
