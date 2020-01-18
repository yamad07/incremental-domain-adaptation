import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self, n_class, n_channels, n_features=256):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
                            n_channels, n_features, 4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose2d(
                            n_features, n_features, 8, stride=4, padding=2, bias=False)
        self.conv3 = nn.ConvTranspose2d(
                            n_features, n_features, 8, stride=4, padding=2, bias=False)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        return F.log_softmax(h, dim=1)
