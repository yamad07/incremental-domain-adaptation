import torch.nn as nn


class IncrementalAdversarialModel(nn.Module):

    def __init__(
            self,
            classifier,
            domain_discriminator,
            source_generator,
            source_discriminator,
            source_encoder,
            target_encoder,
            ):
        super(IncrementalAdversarialModel, self).__init__()
        self.classifier = classifier
        self.domain_discriminator = domain_discriminator
        self.source_generator = source_generator
        self.source_discriminator = source_discriminator
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder

    def forward(self, target_data):

        target_features = self.target_encoder(target_data)
        return self.classifier(target_features)
