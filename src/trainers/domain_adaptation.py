import torch
import torch.nn.functional as F
import numpy as np
from ..analyzers import (
        TargetFeatureVisualizer,
        SourceEncoderAccuracyValidator,
        TargetEncoderAccuracyValidator,
        SourceGeneratorAccuracyValidator
        )


class DATrainerComponent:

    def train(self, epoch, trainer):
        self.trainer = trainer
        self.train_batch_size = self.trainer.train_data_loader.batch_size
        self.valid_batch_size = self.trainer.validate_data_loader.batch_size
        target_validator = TargetEncoderAccuracyValidator()
        source_encoder_validator = SourceEncoderAccuracyValidator()
        source_generator_validator = SourceEncoderAccuracyValidator()

        for e in range(epoch):
            self.trainer.model.target_encoder.train()
            self.trainer.model.source_encoder.train()
            self.trainer.model.source_generator.train()
            self.trainer.experiment.log_current_epoch(e)

            self.trainer.train_data_loader.dataset.train()
            for i, (data, labels) in enumerate(self.trainer.train_data_loader):
                data = data.to(self.trainer.device)
                labels = labels.to(self.trainer.device)

                discriminator_loss = self._ad_train_discriminator(data)
                target_adversarial_loss = self._ad_train_target_encoder(data)

                self.trainer.experiment.log_metric('discriminator_loss', discriminator_loss.item())
                self.trainer.experiment.log_metric('target_adversarial_loss', target_adversarial_loss.item())

            target_validator.validate(trainer)
            source_encoder_validator.validate(trainer)
            source_generator_validator.validate(trainer)

            print("Epoch: {0} D(x): {1} D(G(x)): {2}".format(
                e, discriminator_loss.item(), target_adversarial_loss.item()))

    def _ad_train_target_encoder(self, target_data):
        batch_size = target_data.size(0)
        # init
        self.trainer.target_optim.zero_grad()
        self.trainer.source_optim.zero_grad()
        self.trainer.discrim_optim.zero_grad()

        # forward
        target_features = self.trainer.model.target_encoder(target_data)
        target_features = target_features.view(batch_size, -1)
        target_domain_predicts = self.trainer.model.domain_discriminator(target_features)

        source_domain_labels = torch.ones(target_domain_predicts.size(0))
        source_domain_labels = source_domain_labels.long().to(self.trainer.device)

        target_adversarial_loss = F.nll_loss(target_domain_predicts, source_domain_labels)

        # backward
        target_adversarial_loss.backward()
        self.trainer.target_optim.step()
        return target_adversarial_loss

    def _ad_train_discriminator(self, target_data):

        # init
        self.trainer.target_optim.zero_grad()
        self.trainer.source_optim.zero_grad()
        self.trainer.discrim_optim.zero_grad()

        # forward
        batch_size = target_data.size(0)
        z = torch.randn(batch_size, self.trainer.model.source_generator.z_dim)
        z = z.to(self.trainer.device)
        source_features = self.trainer.model.source_generator(z)
        source_features = source_features.view(batch_size, -1)
        source_domain_preds = self.trainer.model.domain_discriminator(source_features)

        target_features = self.trainer.model.target_encoder(target_data)
        target_features = target_features.view(batch_size, -1)
        target_domain_preds = self.trainer.model.domain_discriminator(target_features)

        preds = torch.cat((source_domain_preds, target_domain_preds))

        source_domain_labels = torch.ones(source_domain_preds.size(0)).long()
        target_domain_labels = torch.zeros(target_domain_preds.size(0)).long()
        labels = torch.cat((source_domain_labels, target_domain_labels)).to(self.trainer.device)

        # backward
        discriminator_loss = F.nll_loss(preds, labels)
        discriminator_loss.backward()
        self.trainer.discrim_optim.step()
        return discriminator_loss

