import torch
import torch.nn.functional as F
import numpy as np
from ..analyzers import GeneratedSourceFeatureVisualizer
from .metrics import Accuracy


class CSMTrainerComponent:

    def train(self, epoch, trainer):
        self.trainer = trainer
        self.train_batch_size = self.trainer.train_data_loader.batch_size
        self.accuracy = Accuracy()

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

            # feature_visualizer.visualize(trainer, e)
            trainer.experiment.log_current_epoch(e)
            print("Epoch: {} Acc: {} D(x): {} D(G(x)): {} NLL F(x): {}".format(
                e, accuracy, discriminator_loss.item(), generator_loss.item(), classification_loss.item(), ))

    def _train_supervised(self, source_data, source_labels):
        # init
        self.trainer.classifier_optim.zero_grad()
        self.trainer.source_optim.zero_grad()

        # forward
        source_features = self.trainer.model.source_encoder(source_data)
        source_preds = self.trainer.model.classifier(source_features)
        classifier_loss = F.nll_loss(source_preds, source_labels)

        # backward
        classifier_loss.backward()

        self.trainer.classifier_optim.step()
        self.trainer.source_optim.step()

        self.trainer.classifier_optim.zero_grad()
        self.trainer.source_optim.zero_grad()

        source_accuracy = self.accuracy(source_preds, source_labels)
        return classifier_loss, source_accuracy

    def _train_source_modeling(self, source_data, source_labels):
        # Train Discriminator
        batch_size = source_data.size(0)
        self.trainer.source_optim.zero_grad()
        self.trainer.source_domain_generator_optim.zero_grad()
        self.trainer.source_domain_discriminator_optim.zero_grad()

        source_features = self.trainer.model.source_encoder(source_data)
        source_features_size = source_features.size()
        source_features = source_features.view(batch_size, -1)
        z = torch.randn(source_labels.size(0), self.trainer.model.source_generator.z_dim)
        z = z.to(self.trainer.device).detach()

        source_fake_features = self.trainer.model.source_generator(z, source_labels)
        true_preds = self.trainer.model.source_discriminator(self._rm_map(source_features.detach()))

        fake_preds = self.trainer.model.source_discriminator(self._rm_map(source_fake_features.detach()))
        preds = torch.cat((true_preds, fake_preds))

        true_labels = torch.ones(true_preds.size(0)).long()
        fake_labels = torch.zeros(fake_preds.size(0)).long()
        labels = torch.cat((true_labels, fake_labels)).to(self.trainer.device)

        discriminator_loss = F.nll_loss(preds, labels)

        discriminator_loss.backward()
        self.trainer.source_domain_discriminator_optim.step()

        # Train Generator and Encoder
        self.trainer.source_optim.zero_grad()
        self.trainer.source_domain_generator_optim.zero_grad()
        self.trainer.source_domain_discriminator_optim.zero_grad()

        z = torch.randn(source_labels.size(0), self.trainer.model.source_generator.z_dim)
        z = z.to(self.trainer.device).detach()
        source_fake_features = self.trainer.model.source_generator(z, source_labels)

        fake_preds = self.trainer.model.source_discriminator(self._rm_map(source_fake_features))
        true_labels = torch.ones(fake_preds.size(0)).long().to(self.trainer.device)
        generator_loss = F.nll_loss(fake_preds, true_labels)
        source_preds = self.trainer.model.classifier(source_fake_features)
        classification_loss = F.nll_loss(source_preds, source_labels)

        loss = generator_loss + classification_loss
        loss.backward()
        self.trainer.source_domain_generator_optim.step()

        return discriminator_loss, generator_loss

    def _rm_map(self, features):

        rm_classifiers = self.trainer.model.randomized_g(self.trainer.model.classifier(features))
        rm_features = self.trainer.model.randomized_f(features)
        features = torch.mul(rm_classifiers, rm_features) / np.sqrt(self.trainer.model.n_random_features)
        return features
