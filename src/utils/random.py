import numpy as np
import torch


def random_labels(n_classes, batch_size):
    labels = np.random.rand(batch_size) * n_classes
    labels = torch.Tensor(np.floor(labels)).long()
    return labels


