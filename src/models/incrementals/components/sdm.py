import torch.nn as nn
import torch.nn.functional as F
from ...layers.gan import fc_layer


class SDMG(nn.Module):

    def __init__(self):
        super(SDMG, self).__init__()
        self.fc1 = fc_layer(100 + 10, 1024)
        self.fc2 = fc_layer(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        h = F.tanh(self.dropout(self.fc4(h)))
        return h


class SDMD(nn.Module):

    def __init__(self):
        super(SDMD, self).__init__()
        self.fc1 = fc_layer(4000, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        return F.softmax(h, dim=1)
