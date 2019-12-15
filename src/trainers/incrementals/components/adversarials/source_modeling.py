import torch
import torch.nn.functional as F


class IncrementalAdversarialSourceModelingTrainerComponent:

    def __init__(self, trainer):
        self.trainer = trainer

    def train(epoch):
        for e in range(sm_epoch):
            for i, (source_data, source_labels, target_data) in enumerate(
                    self.train_data_loader):
                source_data = source_data.to(self.device)
                discriminator_loss, generator_loss = self._train_source_modeling(source_data)
                self.experiment.log_metric('D(x)', discriminator_loss.item())
                self.experiment.log_metric('D(G(x))', generator_loss.item())

            self.experiment.log_current_epoch(e)
            print("Epoch: {0} D(x): {1} D(G(x)): {2}".format(
                e, discriminator_loss.item(), generator_loss.item()))
        self.trainer.model.target_encoder.load_state_dict(
            self.trainer.model.source_encoder.state_dict())
        self.trainer.model.source_generator.eval()

    def _train_source_modeling(self, source_data):
        self.trainer.source_optim.zero_grad()
        self.trainer.source_domain_generator_optim.zero_grad()
        self.trainer.source_domain_discriminator_optim.zero_grad()

        source_features = self.trainer.model.source_encoder(source_data)
        z = torch.randn(16, 100).to(self.device).detach()

        source_fake_features = self.trainer.model.source_generator(z)

        true_preds = self.trainer.model.source_discriminator(
            source_features.detach())
        fake_preds = self.trainer.model.source_discriminator(
            source_fake_features.detach())
        labels = torch.cat(
            (torch.ones(16).long().to(
                self.device), torch.zeros(16).long().to(
                self.device)))
        preds = torch.cat((true_preds, fake_preds))
        discriminator_loss = F.nll_loss(preds, labels)

        discriminator_loss.backward()
        self.trainer.source_domain_discriminator_optim.step()

        self.trainer.source_domain_generator_optim.zero_grad()
        self.trainer.source_domain_discriminator_optim.zero_grad()

        z = torch.randn(16, 100).to(self.device).detach()
        source_fake_features = self.trainer.model.source_generator(z)
        fake_preds = self.trainer.model.source_discriminator(source_fake_features)
        generator_loss = - F.nll_loss(fake_preds,
                                      torch.zeros(16).long().to(self.device))

        generator_loss.backward()
        self.trainer.source_domain_generator_optim.step()

        return discriminator_loss, generator_loss
