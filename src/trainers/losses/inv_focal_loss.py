import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseFocalLoss(nn.Module):

    '''
    input: (batch_size, n_classes)
    targets: (batch_size)
    '''

    def forward(self, input, targets):
        device = input.get_device()
        one_hot = torch.zeros((targets.size()[0], 2)).to(device)
        one_hot = one_hot.scatter_(1, torch.unsqueeze(targets, 1), 1)
        liklyhood = torch.mul(torch.exp(input), one_hot).sum(dim=1)
        return - torch.mean(torch.mul(torch.exp(liklyhood),
                                      torch.log(liklyhood)))
