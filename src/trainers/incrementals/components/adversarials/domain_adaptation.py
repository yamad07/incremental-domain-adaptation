import torch
import torch.nn.functional as F


class IADATrainerComponent:

    def train(self, epoch, trainer):
        self.trainer = trainer
        self.train_batch_size = self.trainer.train_data_loader.batch_size
        self.valid_batch_size = self.trainer.validate_data_loader.batch_size
        trainer.model.target_encoder.load_state_dict(
            trainer.model.source_encoder.state_dict())
        for e in range(epoch):

            for i, (source_data, source_labels, target_data) in enumerate(
                    self.trainer.train_data_loader):
                source_data = source_data.to(self.trainer.device)
                source_labels = source_labels.to(self.trainer.device)
                target_data = target_data.to(self.trainer.device)

                discriminator_loss = self._ad_train_discriminator(
                    source_data, target_data)
                target_adversarial_loss = self._ad_train_target_encoder(
                    target_data)
                self.trainer.experiment.log_metric(
                    'discriminator_loss', discriminator_loss.item())
                self.trainer.experiment.log_metric(
                    'target_adversarial_loss', target_adversarial_loss.item())

            target_valid_accuracy = self._validate(e)
            self.trainer.experiment.log_current_epoch(e)
            self.trainer.experiment.log_metric(
                'valid_target_accuracy',
                target_valid_accuracy)

            print("Epoch: {0} D(x): {1} D(G(x)): {2} target_accuracy: {3}".format(
                e, discriminator_loss.item(), target_adversarial_loss.item(), target_valid_accuracy))

    def _ad_train_target_encoder(self, target_data):
        # init
        self.trainer.target_optim.zero_grad()
        self.trainer.source_optim.zero_grad()
        self.trainer.discrim_optim.zero_grad()

        # forward
        target_features = self.trainer.model.target_encoder(target_data)
        target_domain_predicts = self.trainer.model.domain_discriminator(
            target_features)
        source_labels = torch.ones(target_domain_predicts.size(0)).long().to(self.trainer.device)
        target_adversarial_loss = F.nll_loss(target_domain_predicts, source_labels)

        # backward
        target_adversarial_loss.backward()
        self.trainer.target_optim.step()
        return target_adversarial_loss

    def _ad_train_discriminator(self, source_data, target_data):
        # init
        self.trainer.target_optim.zero_grad()
        self.trainer.source_optim.zero_grad()
        self.trainer.discrim_optim.zero_grad()

        # forward
        z = torch.randn(self.train_batch_size, self.trainer.model.source_generator.z_dim).to(self.trainer.device)
        source_features = self.trainer.model.source_generator(z)
        source_domain_preds = self.trainer.model.domain_discriminator(
            source_features.detach())

        target_features = self.trainer.model.target_encoder(target_data)
        target_domain_preds = self.trainer.model.domain_discriminator(
            target_features.detach())

        domain_labels = torch.cat(
            (torch.ones(source_domain_preds.size(0)).long().to(
                self.trainer.device), torch.zeros(target_domain_preds.size(0)).long().to(
                self.trainer.device)))

        # backward
        discriminator_loss = F.nll_loss(
            torch.cat(
                (source_domain_preds,
                 target_domain_preds)),
            domain_labels)
        discriminator_loss.backward()
        self.trainer.discrim_optim.step()
        return discriminator_loss

    def _validate(self, e):
        accuracy = 0
        for i, (target_data, target_labels) in enumerate(
                self.trainer.validate_data_loader):
            target_data = target_data.to(self.trainer.device)
            target_labels = target_labels.to(self.trainer.device)

            self.trainer.model.classifier.eval()

            target_features = self.trainer.model.target_encoder(target_data)
            target_preds = self.trainer.model.classifier(target_features)
            _, target_preds = torch.max(target_preds, 1)
            accuracy += 100 * \
                (target_preds == target_labels).sum().item() / target_preds.size()[0]

        accuracy /= len(self.trainer.validate_data_loader)
        return accuracy
