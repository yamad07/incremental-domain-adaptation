import torch.nn.functional as F


class IncrementalAdversarialDomainAdaptationTrainerComponent:

    def __init__(self, trainer):
        self.trainer = trainer

    def train(epoch):
        for e in range(epoch):

            for i, (source_data, source_labels, target_data) in enumerate(
                    self.train_data_loader):
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)
                target_data = target_data.to(self.device)

                discriminator_loss = self._ad_train_discriminator(
                    source_data, target_data)
                target_adversarial_loss = self._ad_train_target_encoder(
                    target_data)

                target_features = self.model.target_encoder(target_data)
                target_preds = self.model.classifier(target_features)
                self.experiment.log_metric(
                    'discriminator_loss', discriminator_loss.item())
                self.experiment.log_metric(
                    'target_adversarial_loss', target_adversarial_loss.item())

            target_valid_accuracy = self.validate(e)
            self.experiment.log_current_epoch(e)
            self.experiment.log_metric(
                'valid_target_accuracy',
                target_valid_accuracy)

            print("Epoch: {0} D(x): {1} D(G(x)): {2} target_accuracy: {3}".format(
                e, discriminator_loss.item(), target_adversarial_loss.item(), target_valid_accuracy))

    def _ad_train_target_encoder(self, target_data):
        # init
        self.target_optim.zero_grad()
        self.source_optim.zero_grad()
        self.discrim_optim.zero_grad()

        # forward
        target_features = self.model.target_encoder(target_data)
        target_domain_predicts = self.model.domain_discriminator(
            target_features)
        target_adversarial_loss = - \
            F.nll_loss(target_domain_predicts, torch.zeros(16).long().to(self.device))

        # backward
        target_adversarial_loss.backward()
        self.target_optim.step()
        return target_adversarial_loss

    def _ad_train_discriminator(self, source_data, target_data):
        # init
        self.target_optim.zero_grad()
        self.source_optim.zero_grad()
        self.discrim_optim.zero_grad()

        # forward
        z = torch.randn(16, 100).to(self.device)
        source_features = self.model.source_generator(z)
        # source_features = self.source_encoder(source_data)
        source_domain_preds = self.model.domain_discriminator(
            source_features.detach())

        target_features = self.model.target_encoder(target_data)
        target_domain_preds = self.model.domain_discriminator(
            target_features.detach())

        domain_labels = torch.cat(
            (torch.ones(16).long().to(
                self.device), torch.zeros(16).long().to(
                self.device)))

        # backward
        discriminator_loss = F.nll_loss(
            torch.cat(
                (source_domain_preds,
                 target_domain_preds)),
            domain_labels)
        discriminator_loss.backward()
        self.discrim_optim.step()
        return discriminator_loss
