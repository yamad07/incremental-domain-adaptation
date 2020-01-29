import torch.nn as nn


class KLDivergence(nn.Module):

    def forward(self, x1, x2):
        mean_x1 = x1.mean()
        mean_x2 = x2.mean()
        std_x1 = x1.std() + 1e-5
        std_x2 = x2.std() + 1e-5

        return (std_x1 / std_x2).log() + (std_x2 ** 2 + (mean_x2 - mean_x1) **2 ) /  (2 * std_x1 ** 2) - (1 / 2)
