import torch.nn as nn


class NetworkInNetworkLayer(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super(NetworkInNetworkLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=int(
                    (kernel_size - 1) / 2)),
            nn.ELU(
                inplace=True),
            nn.Conv2d(
                out_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=int(
                    (kernel_size - 1) / 2)),
            nn.ELU(
                inplace=True),
            nn.Conv2d(
                out_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=int(
                    (kernel_size - 1) / 2)),
            nn.ELU(
                inplace=True),
            nn.BatchNorm2d(out_dim),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2),
        )

    def forward(self, x):
        return self.layer(x)
