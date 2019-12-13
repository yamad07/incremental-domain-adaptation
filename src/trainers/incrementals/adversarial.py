import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class IncrementalAdversarialTrainer:

    def __init__(self,
            model,
            experiment,
            train_data_loader,
            valid_data_loader,
            cuda_id
            ):
        self.model = model
        self.experiment = experiment

        self.train_data_loader = train_data_loader
        self.validate_data_loader = valid_data_loader
        self.device = torch.device("cuda:{}".format(int(cuda_id)) if torch.cuda.is_available() else "cpu")

        self.classifier_optim = optim.SGD(self.model.classifier.parameters(), lr=1e-3)
        self.source_optim = optim.Adam(self.model.source_encoder.parameters(), lr=1e-3)
        self.target_optim = optim.Adam(self.model.target_encoder.parameters(), lr=1e-4)
        self.discrim_optim = optim.Adam(self.model.domain_discriminator.parameters(), lr=1e-4)
        self.source_domain_discriminator_optim = optim.Adam(self.model.source_discriminator.parameters(), lr=1e-4)
        self.source_domain_generator_optim = optim.Adam(self.model.source_generator.parameters(), lr=1e-4)


    def train(self, s_epoch, sm_epoch, da_epoch):

        self.model.to(self.device)
        self.model.train()
        for e in range(s_epoch):
            for i, (source_data, source_labels, target_data) in enumerate(self.train_data_loader):
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)
                target_data = target_data.to(self.device)
                classifier_loss, source_accuracy = self._train_source(source_data, source_labels)

            self.experiment.log_current_epoch(e)
            self.experiment.log_metric('source_accuracy', source_accuracy)
            print("Epoch: {0} classifier: {1} source accuracy: {2}".format(e, classifier_loss, source_accuracy))


        for e in range(sm_epoch):
            for i, (source_data, source_labels, target_data) in enumerate(self.train_data_loader):
                source_data = source_data.to(self.device)
                discriminator_loss, generator_loss = self._train_source_modeling(source_data)
                self.experiment.log_metric('D(x)', discriminator_loss)
                self.experiment.log_metric('D(G(x))', generator_loss)

            self.experiment.log_current_epoch(e)
            print("Epoch: {0} D(x): {1} D(G(x)): {2}".format(e, discriminator_loss, generator_loss))

        self.model.target_encoder.load_state_dict(self.model.source_encoder.state_dict())
        self.model.source_generator.eval()

        for e in range(epoch):

            for i, (source_data, source_labels, target_data) in enumerate(self.train_data_loader):
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)
                target_data = target_data.to(self.device)

                discriminator_loss = self._ad_train_discriminator(source_data, target_data)
                target_adversarial_loss = self._ad_train_target_encoder(target_data)

                target_features = self.target_encoder(target_data)
                target_preds = self.classifier(target_features)
                self.experiment.log_metric('discriminator_loss', discriminator_loss)
                self.experiment.log_metric('target_adversarial_loss', target_adversarial_loss)

            target_valid_accuracy = self.validate(e)
            self.experiment.log_current_epoch(e)
            self.experiment.log_metric('valid_target_accuracy', target_valid_accuracy)

            print("Epoch: {0} D(x): {1} D(G(x)): {2} target_accuracy: {3}".format(
                e, discriminator_loss, target_adversarial_loss, target_valid_accuracy))

    def adaptation(self, da_epoch):

        for e in range(da_epoch):

            for i, (source_data, source_labels, target_data) in enumerate(self.data_loader):
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)
                target_data = target_data.to(self.device)

                discriminator_loss = self._ad_train_discriminator(source_data, target_data)
                target_adversarial_loss = self._ad_train_target_encoder(target_data)

                target_features = self.target_encoder(target_data)
                target_preds = self.classifier(target_features)
                self.experiment.log_metric('discriminator_loss', discriminator_loss)
                self.experiment.log_metric('target_adversarial_loss', target_adversarial_loss)

            target_valid_accuracy = self.validate(e)
            self.experiment.log_current_epoch(e)
            self.experiment.log_metric('valid_target_accuracy', target_valid_accuracy)

            print("Epoch: {0} D(x): {1} D(G(x)): {2} target_accuracy: {3}".format(
                e, discriminator_loss, target_adversarial_loss, target_valid_accuracy))


    def validate(self):
        accuracy = 0
        for i, (target_data, target_labels) in enumerate(self.validate_data_loader):
            target_data = target_data.to(self.device)
            target_labels = target_labels.to(self.device)

            self.model.target_encoder.eval()
            self.model.classifier.eval()

            target_features = self.target_encoder(target_data)
            target_preds = self.classifier(target_features)
            _, target_preds = torch.max(target_preds, 1)
            accuracy += 100 * (target_preds == target_labels).sum().item() / target_preds.size()[0]

        accuracy /= len(self.validate_data_loader)
        return accuracy

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_validate_data_loader(self, validate_data_loader):
        self.validate_data_loader = validate_data_loader

    def _train_source(self, source_data, source_labels):
        # init
        self.classifier_optim.zero_grad()
        self.source_optim.zero_grad()

        # forward
        source_features = self.model.source_encoder(source_data)
        source_preds = self.model.classifier(source_features)
        classifier_loss = F.nll_loss(source_preds, source_labels)

        # backward
        classifier_loss.backward()

        self.classifier_optim.step()
        self.source_optim.step()
        source_accuracy = self._calc_accuracy(source_preds, source_labels)
        return classifier_loss, source_accuracy

    def _train_source_modeling(self, source_data):
        self.source_optim.zero_grad()
        self.source_domain_generator_optim.zero_grad()
        self.source_domain_discriminator_optim.zero_grad()

        source_features = self.model.source_encoder(source_data)
        z = torch.randn(16, 100).to(self.device).detach()

        source_fake_features = self.model.source_generator(z)

        true_preds = self.model.source_domain_discriminator(source_features.detach())
        fake_preds = self.model.source_domain_discriminator(source_fake_features.detach())
        labels = torch.cat((torch.ones(16).long().to(self.device), torch.zeros(16).long().to(self.device)))
        preds = torch.cat((true_preds, fake_preds))
        discriminator_loss = F.nll_loss(preds, labels)

        discriminator_loss.backward()
        self.source_domain_discriminator_optim.step()

        self.source_domain_generator_optim.zero_grad()
        self.source_domain_discriminator_optim.zero_grad()

        z = torch.randn(16, 100).to(self.device).detach()
        source_fake_features = self.source_generator(z)
        fake_preds = self.source_domain_discriminator(source_fake_features)
        generator_loss = - F.nll_loss(fake_preds, torch.zeros(16).long().to(self.device))

        generator_loss.backward()
        self.source_domain_generator_optim.step()

        return discriminator_loss, generator_loss


    def _ad_train_target_encoder(self, target_data):
        # init
        self.target_optim.zero_grad()
        self.source_optim.zero_grad()
        self.discrim_optim.zero_grad()

        # forward
        target_features = self.model.target_encoder(target_data)
        target_domain_predicts = self.model.domain_discriminator(target_features)
        target_adversarial_loss = - F.nll_loss(target_domain_predicts, torch.zeros(16).long().to(self.device))

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
        source_domain_preds = self.model.domain_discriminator(source_features.detach())

        target_features = self.model.target_encoder(target_data)
        target_domain_preds = self.model.domain_discriminator(target_features.detach())

        domain_labels = torch.cat((torch.ones(16).long().to(self.device), torch.zeros(16).long().to(self.device)))

        # backward
        discriminator_loss = F.nll_loss(torch.cat((source_domain_preds, target_domain_preds)), domain_labels)
        discriminator_loss.backward()
        self.discrim_optim.step()
        return discriminator_loss

    def _calc_accuracy(self, preds, labels):
        _, preds = torch.max(preds, 1)
        accuracy = 100 * (preds == labels).sum().item() / preds.size()[0]
        return accuracy