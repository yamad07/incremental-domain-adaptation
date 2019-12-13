import torch.nn as nn


class RandomizedMultilinear(nn.Module):

    def __init__(self, n_input):
        super(RandomizedMultilinear, self).__init__()
        self.f = nn.Linear(n_input, 4000)

    def forward(self, input):
        return self.f(input)
