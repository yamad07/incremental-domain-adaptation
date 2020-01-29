import torch
import torch.nn.functional as F
from typing import Tuple, List
import math


class AdaINTransfer:

    def forward(
            self,
            content_features: torch.FloatTensor,
            style_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        eps = 1e-5

        size = content_features.size()
        batch_size, n_channels = size[0], size[1]
        content_features_flatten = content_features.view(n_channels, -1)

        content_mean = content_features_flatten.mean(1)
        content_mean = content_mean.view(1, n_channels, 1, 1)

        content_std = content_features_flatten.std(1)
        content_std = content_std.view(1, n_channels, 1, 1) + eps

        style_features_flatten = style_features.view(style_features.size(1), -1)

        style_mean = style_features_flatten.mean(1)
        style_mean = style_mean.view(1, style_features.size(1), 1, 1)

        style_std = style_features_flatten.std(1)
        style_std = style_std.view(1, style_features.size(1), 1, 1) + eps

        return ((content_features - content_mean.expand(size)) / \
                content_std.expand(size)) * style_std.expand(size) + style_mean.expand(size)

class AdaLNTransfer:

    def forward(
            self,
            content_features: torch.FloatTensor,
            style_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        eps = 1e-5

        size = content_features.size()
        style_size = style_features.size()
        batch_size, n_channels, w = size[0], size[1], size[2]
        
        content_features_flatten = content_features.view(
            batch_size, n_channels, -1)

        content_mean = content_features_flatten.mean(1)
        content_mean = content_mean.view(batch_size, 1, w, w)

        content_std = content_features_flatten.std(1)
        content_std = content_std.view(batch_size, 1, w, w) + eps

        style_features_flatten = style_features.view(
            style_features.size(0), style_features.size(1), -1)

        style_mean = style_features_flatten.mean(1)
        style_mean = style_mean.view(style_features.size(0), 1, w, w)

        style_std = style_features_flatten.std(1)
        style_std = style_std.view(style_features.size(0), 1, w, w) + eps

        return ((content_features - content_mean.expand(size)) / \
                content_std.expand(size)) * style_std.expand(style_size) + style_mean.expand(style_size)