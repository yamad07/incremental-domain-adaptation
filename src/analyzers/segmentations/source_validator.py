import torch
from ..trainers.metrics import Accuracy


class SourceEncoderAccuracyValidator:

    def __init__(self, analyzer_dir='results'):
        self.accuracy = Accuracy()

    def validate(self, trainer):
        target_preds_batch = []
        target_labels_batch = []
        trainer.model.source_encoder.eval()
        trainer.model.classifier.eval()

        for i, (target_data, target_labels) in enumerate(trainer.validate_data_loader):
            target_data = target_data.to(trainer.device)
            target_labels = target_labels.to(trainer.device)


            target_features = trainer.model.source_encoder(target_data)
            target_preds = trainer.model.classifier.prob(target_features)

            target_preds_batch.append(target_preds)
            target_labels_batch.append(target_labels)

        target_preds_batch = torch.cat(target_preds_batch, dim=0)
        target_labels_batch = torch.cat(target_labels_batch, dim=0)

        target_accuracy = self.accuracy(target_preds_batch, target_labels_batch)
        trainer.experiment.log_metric('valid_source_encoder_target_accuracy', target_accuracy)
        return target_accuracy

