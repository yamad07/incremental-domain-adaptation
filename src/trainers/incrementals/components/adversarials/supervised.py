import torch.nn.functional as F


class IncrementalAdversarialSupervisedTrainerComponent:

    def __init__(self, trainer):
        self.trainer = trainer

    def train(epoch):
        for e in range(epoch):
            for i, (source_data, source_labels, target_data) in enumerate(
                    self.train_data_loader):
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)
                target_data = target_data.to(self.device)
                classifier_loss, source_accuracy = self._train_source(
                    source_data, source_labels)

            self.experiment.log_current_epoch(e)
            self.experiment.log_metric('source_accuracy', source_accuracy)
            print(
                "Epoch: {0} classifier: {1} source accuracy: {2}".format(
                    e, classifier_loss, source_accuracy))

    def _train_source(self, source_data, source_labels):
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
        source_accuracy = self._calc_accuracy(source_preds, source_labels)
        return classifier_loss, source_accuracy
