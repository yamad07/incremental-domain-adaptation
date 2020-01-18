import torch
from ..trainers.metrics import Accuracy


class TargetEncoderAccuracyValidator:

    def __init__(self, analyzer_dir='results'):
        self.accuracy = Accuracy()

    def validate(self, trainer):
        target_preds_batch = []
        target_labels_batch = []
        trainer.model.target_encoder.eval()
        trainer.model.classifier.eval()
        trainer.validate_data_loader.dataset.eval()
        with torch.no_grad():
            for i, (target_data, target_labels) in enumerate(trainer.validate_data_loader):
                target_data = target_data.to(trainer.device)
                target_labels = target_labels.to(trainer.device)

                target_features = trainer.model.target_encoder(target_data)
                target_preds = trainer.model.classifier(target_features).detach()

                target_preds_batch.append(target_preds.cpu())
                target_labels_batch.append(target_labels.cpu())

                trainer.model.target_encoder.zero_grad()
                trainer.model.classifier.zero_grad()

        target_preds_batch = torch.cat(target_preds_batch, dim=0)
        target_labels_batch = torch.cat(target_labels_batch, dim=0)

        target_accuracy = self.accuracy(target_preds_batch, target_labels_batch)
        trainer.experiment.log_metric('{}_valid_target_accuracy'.format(trainer.validate_data_loader.dataset.domain_name), target_accuracy)
        trainer.model.target_encoder.train()
        trainer.model.classifier.train()
        return target_accuracy

