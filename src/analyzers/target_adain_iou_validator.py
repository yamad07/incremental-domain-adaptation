import torch
import torch.nn.functional as F
from ..trainers.metrics import Accuracy
from ..utils.adain import AdaINTransfer
from sklearn.metrics import confusion_matrix


class TargetEncoderAdaINAccuracyValidator:

    def __init__(self, analyzer_dir='results'):
        self.accuracy = Accuracy()
        self.transfer = AdaINTransfer()

    def validate(self, trainer):
        target_preds_batch = []
        target_labels_batch = []
        trainer.model.source_encoder.eval()
        trainer.model.classifier.eval()
        trainer.model.source_generator.eval()
        trainer.validate_data_loader.dataset.eval()
        with torch.no_grad():
            for i, (target_data, target_labels) in enumerate(trainer.validate_data_loader):
                batch_size = target_data.size(0)
                target_data = target_data.to(trainer.device)
                target_labels = target_labels.to(trainer.device)
                target_features = trainer.model.source_encoder(target_data)
                z = torch.randn(batch_size, trainer.model.source_generator.z_dim)
                z = z.to(trainer.device)
                source_features = trainer.model.source_generator(z).detach()
                size = target_features.size()
                source_features = source_features.view(size)
                target_features = self.transfer.forward(target_features, source_features)

                b = target_features.size(0)
                target_features = target_features.view(b, -1)
                target_preds = trainer.model.classifier(target_features).detach()

                target_preds_batch.append(target_preds.cpu())
                target_labels_batch.append(target_labels.cpu())

        target_preds_batch = torch.cat(target_preds_batch, dim=0).numpy()
        target_labels_batch = torch.cat(target_labels_batch, dim=0).numpy()
        conf_matrix = confusion_matrix(target_preds_batch, target_labels_batch)
        tp = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        iou = tp / (tp + fp + fn)

        trainer.experiment.log_metric('{}_valid_target_iou'.format(trainer.validate_data_loader.dataset.domain_name), iou)
        trainer.model.target_encoder.train()
        trainer.model.classifier.train()
        return target_accuracy

