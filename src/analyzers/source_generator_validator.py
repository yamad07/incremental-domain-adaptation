import torch
from ..trainers.metrics import Accuracy


class SourceGeneratorAccuracyValidator:

    def __init__(self, analyzer_dir='results'):
        self.accuracy = Accuracy()

    def validate(self, trainer):
        source_preds_batch = []
        source_labels_batch = []
        with torch.no_grad():
            for i, (_, source_labels, _) in enumerate(trainer.train_data_loader):
                source_labels = source_labels.to(trainer.device)

                z = torch.randn(source_labels.size(0), trainer.model.source_generator.z_dim)
                z = z.to(trainer.device)

                source_preds = trainer.model.classifier.prob(trainer.model.source_generator(z, source_labels))
                source_preds_batch.append(source_preds)
                source_labels_batch.append(source_labels)

        source_preds_batch = torch.cat(source_preds_batch, dim=0)
        source_labels_batch = torch.cat(source_labels_batch, dim=0)

        source_accuracy = self.accuracy(source_preds_batch, source_labels_batch)
        trainer.experiment.log_metric('source_generator_accuracy', source_accuracy)
        return source_accuracy

