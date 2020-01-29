from src.datasets import (
        IDAMNIST,
        IDASVHN
        )
import torch.utils.data as data
from torchvision import utils as vutils

svhn_dataset = IDASVHN(
    root='./data/',
    download=True,
    )
train_data_loader = data.DataLoader(svhn_dataset, batch_size=16, shuffle=True)

for data, _ in train_data_loader:
    vutils.save_image(data, "svhn.jpg", normalize=True)
    break
