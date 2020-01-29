import torch
import torch.nn.functional as F
from ..trainers.metrics import Accuracy
from ..utils import KLDivergence


class TargetKLDValidator:

    def __init__(self, analyzer_dir='results'):
        self.accuracy = Accuracy()
        self.divergence = KLDivergence()

    def validate(self, trainer):
        target_preds_batch = []
        target_labels_batch = []

        trainer.model.source_generator.eval()
        trainer.model.source_encoder.eval()
        trainer.model.classifier.eval()
        trainer.validate_data_loader.dataset.eval()
        kl_divs = []
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

                b = target_features.size(0)
                kl_divs.append(self.divergence(source_features, target_features))

        kl_divs = torch.Tensor(kl_divs).mean()
        trainer.experiment.log_metric('{}_valid_kld'.format(trainer.validate_data_loader.dataset.domain_name), kl_divs.item())
        trainer.model.target_encoder.train()
        trainer.model.classifier.train()
        return kl_divs

