import torch
import torch.nn.functional as F


class IASMTrainerComponent:

    def train(self, epoch, trainer):
        self.trainer = trainer
        self.train_batch_size = self.trainer.train_data_loader.batch_size
        self.valid_batch_size = self.trainer.validate_data_loader.batch_size
        for e in range(epoch):
            for i, (source_data, source_labels, target_data) in enumerate(
                    trainer.train_data_loader):
                source_data = source_data.to(trainer.device)
                discriminator_loss, generator_loss = self._train_source_modeling(source_data)
                trainer.experiment.log_metric('D(x)', discriminator_loss.item())
                trainer.experiment.log_metric('D(G(x))', generator_loss.item())

            trainer.experiment.log_current_epoch(e)
            print("Epoch: {0} D(x): {1} D(G(x)): {2}".format(
                e, discriminator_loss.item(), generator_loss.item()))

    def _train_source_modeling(self, source_data):
        self.trainer.source_optim.zero_grad()
        self.trainer.source_domain_generator_optim.zero_grad()
        self.trainer.source_domain_discriminator_optim.zero_grad()

        source_features = self.trainer.model.source_encoder(source_data)
        z = torch.randn(self.train_batch_size, self.trainer.model.source_generator.z_dim)
        z = z.to(self.trainer.device).detach()

        source_fake_features = self.trainer.model.source_generator(z)

        true_preds = self.trainer.model.source_discriminator(
            source_features.detach())
        fake_preds = self.trainer.model.source_discriminator(
            source_fake_features.detach())

        true_labels = torch.ones(source_features.size(0)).long()
        true_labels = true_labels.to(self.trainer.device)
        fake_labels = torch.zeros(source_features.size(0)).long()
        fake_labels = fake_labels.to(self.trainer.device)
        labels = torch.cat((true_labels, fake_labels))
        preds = torch.cat((true_preds, fake_preds))
        discriminator_loss = F.nll_loss(preds, labels)

        discriminator_loss.backward()
        self.trainer.source_domain_discriminator_optim.step()

        self.trainer.source_domain_generator_optim.zero_grad()
        self.trainer.source_domain_discriminator_optim.zero_grad()

        z = torch.randn(self.train_batch_size, self.trainer.model.source_generator.z_dim)
        z = z.to(self.trainer.device).detach()
        source_fake_features = self.trainer.model.source_generator(z)
        fake_preds = self.trainer.model.source_discriminator(source_fake_features)
        true_labels = torch.ones(source_features.size(0)).long()
        true_labels = true_labels.to(self.trainer.device)
        generator_loss = F.nll_loss(fake_preds, true_labels)

        generator_loss.backward()
        self.trainer.source_domain_generator_optim.step()

        return discriminator_loss, generator_loss
