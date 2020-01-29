import torch
from ..trainers.metrics import Accuracy
from sklearn.metrics import confusion_matrix


class TargetEncoderIoUAccuracyValidator:

    def __init__(self, analyzer_dir='results'):
        self.accuracy = Accuracy()

    def validate(self, trainer):
        target_preds_batch = []
        target_labels_batch = []

        trainer.model.source_encoder.eval()
        trainer.model.classifier.eval()
        trainer.train_data_loader.dataset.eval()
        with torch.no_grad():
            for i, (target_data, target_labels) in enumerate(trainer.validate_data_loader):
                target_data = target_data.to(trainer.device)
                target_labels = target_labels.to(trainer.device)

                target_features = trainer.model.target_encoder(target_data)
                target_preds = trainer.model.classifier(target_features).detach()

                target_preds_batch.append(target_preds.cpu())
                target_labels_batch.append(target_labels.cpu())

        target_preds_batch = torch.cat(target_preds_batch, dim=0).max(1)[1].view(-1).numpy()
        target_labels_batch = torch.cat(target_labels_batch, dim=0).view(-1).numpy()

        conf_matrix = confusion_matrix(target_preds_batch, target_labels_batch)
        tp = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        iou = tp / (tp + fp + fn)
        trainer.experiment.log_metric('{}_valid_target_encoder_iou'.format(trainer.validate_data_loader.dataset.domain_name), iou)
        trainer.model.source_encoder.train()
        trainer.model.classifier.train()
        return iou

