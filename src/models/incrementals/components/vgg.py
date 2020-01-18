import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGEncoder(nn.Module):

    def __init__(self):
        super(VGGEncoder, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features

    def features(self, x):
        h = self.vgg(x)
        b = h.size(0)
        return h.view(b, -1)

    def forward(self, x):
        return self.vgg(x)
