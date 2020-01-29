import torch
import torch.nn.functional as F
from typing import Tuple, List


class AdaINTransfer:

    def forward(
            self,
            content_features: torch.FloatTensor,
            style_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        eps = 1e-5

        size = content_features.size()
        batch_size, n_channels = size[0], size[1]
        content_features_flatten = content_features.view(
            batch_size, n_channels, -1)

        content_mean = content_features_flatten.mean(2)
        content_mean = content_mean.view(batch_size, n_channels, 1, 1)

        content_std = content_features_flatten.std(2)
        content_std = content_std.view(batch_size, n_channels, 1, 1) + eps

        style_features_flatten = style_features.view(
            batch_size, n_channels, -1)

        style_mean = style_features_flatten.mean(2)
        style_mean = style_mean.view(batch_size, n_channels, 1, 1)

        style_std = style_features_flatten.std(2)
        style_std = style_std.view(batch_size, n_channels, 1, 1) + eps

        return ((content_features - content_mean.expand(size)) / \
                content_std.expand(size)) * style_std.expand(size) + style_mean.expand(size)
