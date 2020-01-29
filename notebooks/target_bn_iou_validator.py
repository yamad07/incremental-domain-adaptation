import torch
import torch.nn.functional as F
from ..trainers.metrics import Accuracy
from ..utils.adain import AdaINTransfer
from sklearn.metrics import confusion_matrix


class TargetEncoderBNIoUAccuracyValidator:

    def __init__(self, analyzer_dir='results'):
        self.accuracy = Accuracy()
        self.transfer = AdaINTransfer()

    def validate(self, trainer):
        target_preds_batch = []
        target_labels_batch = []

        source_generator.eval()
        source_encoder.eval()
        classifier.eval()
        validate_data_loader.dataset.eval()
        with torch.no_grad():
            for i, (target_data, target_labels) in enumerate(validate_data_loader):
                batch_size = target_data.size(0)
                target_data = target_data.to(trainer.device)
                target_labels = target_labels.to(trainer.device)
                target_features = source_encoder(target_data)
                z = torch.randn(batch_size, source_generator.z_dim)
                z = z.to(trainer.device)
                source_features = source_generator(z).detach()
                size = target_features.size()
                source_features = source_features.view(size)

                b = target_features.size(0)
                target_features = target_features - source_features.mean() / target_features.std() * source_features.std() + source_features.mean()
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
        trainer.experiment.log_metric('{}_valid_target_iou'.format(trainer.validate_data_loader.dataset.domain_name), iou)
        trainer.model.target_encoder.train()
        trainer.model.classifier.train()
        return iou

