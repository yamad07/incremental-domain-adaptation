import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_in_network import NetworkInNetworkLayer


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.c1 = NetworkInNetworkLayer(1, 64, kernel_size=3)
        self.c2 = NetworkInNetworkLayer(64, 64, kernel_size=3)
        self.c3 = NetworkInNetworkLayer(64, 128, kernel_size=3)
        self.c4 = NetworkInNetworkLayer(128, 256, kernel_size=3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h = F.leaky_relu(self.c1(x))
        h = F.leaky_relu(self.c2(h))
        h = F.leaky_relu(self.c3(h))
        h = torch.tanh(self.dropout(self.c4(h)))
        a, b, c, d = h.size()
        h = h.view(a, b * c * d)
        return h
