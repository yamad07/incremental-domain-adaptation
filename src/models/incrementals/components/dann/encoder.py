import torch
import torch.nn as nn
import torch.nn.functional as F


class DANNEncoder(nn.Module):


    def __init__(self):
        super(DANNEncoder, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                # nn.BatchNorm2d(32),
                )

        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                # nn.BatchNorm2d(64),
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # nn.Tanh(),
                )

    def forward(self, img):
        h = self.conv1(img)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        return h
        # b = h.size(0)
        # return h.view(b, -1)
