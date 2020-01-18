import torch


class Accuracy:
    def __call__(self, preds, labels):
        _, preds = torch.max(preds, 1)
        accuracy = 100 * (preds == labels).sum().item() / preds.view(-1).size(0)
        return accuracy

