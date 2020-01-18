import torch
import torch.nn as nn
import torch.nn.functional as F


class DANNEncoder(nn.Module):


    def __init__(self):
        super(DANNEncoder, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.AvgPool2d(kernel_size=2, stride=2)
                )

        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2, stride=2)
                )

    def forward(self, img):
        h = self.conv1(img)
        h = self.conv2(h)
        b = h.size(0)
        h = h.view(b, -1)

        return h
