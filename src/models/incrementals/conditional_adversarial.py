import torch.nn as nn
from .components.cada.randomized_multilinear import RandomizedMultilinear


class IncrementalConditionalAdversarialModel(nn.Module):

    def __init__(
            self,
            classifier,
            domain_discriminator,
            source_generator,
            source_discriminator,
            source_encoder,
            target_encoder,
            n_classes,
            n_features,
            n_random_features,
    ):
        super(IncrementalConditionalAdversarialModel, self).__init__()
        self.classifier = classifier
        self.domain_discriminator = domain_discriminator
        self.source_generator = source_generator
        self.source_discriminator = source_discriminator
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.n_random_features = n_random_features
        self.n_features = n_features
        self.randomized_g = RandomizedMultilinear(n_classes, n_random_features)
        self.randomized_f = RandomizedMultilinear(n_features, n_random_features)

    def forward(self, target_data):

        target_features = self.target_encoder(target_data)
        return self.classifier(target_features)
