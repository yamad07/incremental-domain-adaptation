import torch.nn as nn
import torch.nn.functional as F
from ...layers.gan import fc_layer


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = fc_layer(4000, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = self.fc3(h)
        return F.softmax(h, dim=1)
