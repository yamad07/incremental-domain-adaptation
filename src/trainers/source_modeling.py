import torch
import torch.nn.functional as F
from .losses.inv_focal_loss import InverseFocalLoss
import numpy as np
from ..analyzers import (
        TargetFeatureVisualizer,
        TargetEncoderBNAccuracyValidator,
        SourceEncoderAccuracyValidator,
        TargetEncoderAccuracyValidator,
        SourceGeneratorAccuracyValidator,
        GeneratedSourceFeatureVisualizer
        )
from .metrics import Accuracy


class SMTrainerComponent:

    def train(self, epoch, trainer):
        self.trainer = trainer
        self.train_batch_size = self.trainer.train_data_loader.batch_size
        self.accuracy = Accuracy()
        self.criterion = InverseFocalLoss()

        feature_visualizer = GeneratedSourceFeatureVisualizer()
        for e in range(epoch):
            for i, (source_data, source_labels) in enumerate(
                    trainer.train_data_loader):
                source_data = source_data.to(trainer.device)
                source_labels = source_labels.to(trainer.device)
                classification_loss, accuracy = self._train_supervised(source_data, source_labels)
                discriminator_loss, generator_loss = self._train_source_modeling(source_data, source_labels)

                trainer.experiment.log_metric('D(x)', discriminator_loss.item())
                trainer.experiment.log_metric('D(G(x))', generator_loss.item())
                trainer.experiment.log_metric('NLL F(x)', classification_loss.item())
                trainer.experiment.log_metric('Accuracy', accuracy)

            feature_visualizer.visualize(trainer, e)
            trainer.experiment.log_current_epoch(e)
            print("Epoch: {} Acc: {} D(x): {} D(G(x)): {} NLL F(x): {}".format(
                e, accuracy, discriminator_loss.item(), generator_loss.item(), classification_loss.item(), ))

    def _train_supervised(self, source_data, source_labels):
        # init
        self.trainer.source_optim.zero_grad()
        self.trainer.classifier_optim.zero_grad()
        self.trainer.source_domain_generator_optim.zero_grad()
        self.trainer.source_domain_discriminator_optim.zero_grad()

        # forward
        batch_size = source_data.size(0)
        source_features = self.trainer.model.source_encoder(source_data)
        # source_features = source_features.view(batch_size, -1)
        source_preds = self.trainer.model.classifier(source_features)
        classifier_loss = F.nll_loss(source_preds, source_labels)

        # backward
        classifier_loss.backward()

        self.trainer.classifier_optim.step()
        self.trainer.source_optim.step()

        source_accuracy = self.accuracy(source_preds, source_labels)
        return classifier_loss, source_accuracy

    def _train_source_modeling(self, source_data, source_labels):
        # Train Discriminator
        batch_size = source_data.size(0)
        self.trainer.source_optim.zero_grad()
        self.trainer.classifier_optim.zero_grad()
        self.trainer.source_domain_generator_optim.zero_grad()
        self.trainer.source_domain_discriminator_optim.zero_grad()

        source_features = self.trainer.model.source_encoder(source_data)
        source_features_size = source_features.size()
        source_features = source_features.view(batch_size, -1)

        true_preds = self.trainer.model.source_discriminator(source_features.detach())

        z = torch.randn(source_labels.size(0), self.trainer.model.source_generator.z_dim)
        z = z.to(self.trainer.device).detach()
        source_fake_features = self.trainer.model.source_generator(z)
        source_fake_features = source_fake_features.view(source_fake_features.size(0), -1)
        fake_preds = self.trainer.model.source_discriminator(source_fake_features.detach())

        preds = torch.cat((true_preds, fake_preds))

        true_labels = torch.ones(true_preds.size(0)).long()
        fake_labels = torch.zeros(fake_preds.size(0)).long()
        labels = torch.cat((true_labels, fake_labels)).to(self.trainer.device)

        # discriminator_loss = F.nll_loss(preds, labels)
        discriminator_loss = self.criterion(preds, labels)

        discriminator_loss.backward()
        self.trainer.source_domain_discriminator_optim.step()

        # Train Generator and Encoder
        self.trainer.source_optim.zero_grad()
        self.trainer.classifier_optim.zero_grad()
        self.trainer.source_domain_generator_optim.zero_grad()
        self.trainer.source_domain_discriminator_optim.zero_grad()

        z = torch.randn(source_labels.size(0), self.trainer.model.source_generator.z_dim)
        z = z.to(self.trainer.device).detach()
        source_fake_features = self.trainer.model.source_generator(z)

        source_fake_features = source_fake_features.view(source_fake_features.size(0), -1)
        fake_preds = self.trainer.model.source_discriminator(source_fake_features)
        fake_labels = torch.zeros(fake_preds.size(0)).long().to(self.trainer.device)
        # generator_loss = - F.nll_loss(fake_preds, fake_labels)
        generator_loss = - self.criterion(fake_preds, fake_labels)

        generator_loss.backward()
        self.trainer.source_domain_generator_optim.step()

        return discriminator_loss, generator_loss
