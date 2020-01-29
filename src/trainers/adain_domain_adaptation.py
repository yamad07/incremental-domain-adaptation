import torch
import torch.nn.functional as F
import numpy as np
from ..analyzers import (
        TargetFeatureVisualizer,
        SourceEncoderAccuracyValidator,
        TargetEncoderAdaINAccuracyValidator,
        TargetEncoderBNAccuracyValidator,
        SourceGeneratorAccuracyValidator,
        TargetEncoderBNIoUAccuracyValidator,
        SourceEncoderIoUAccuracyValidator
        )


class AdaINDATrainerComponent:

    def train(self, epoch, trainer):
        self.trainer = trainer
        self.train_batch_size = self.trainer.train_data_loader.batch_size
        self.valid_batch_size = self.trainer.validate_data_loader.batch_size
        target_validator = TargetEncoderBNIoUAccuracyValidator()
        # target_validator = TargetEncoderBNAccuracyValidator()
        source_encoder_validator = SourceEncoderIoUAccuracyValidator()

        for e in range(epoch):
            self.trainer.model.target_encoder.train()
            self.trainer.model.domain_discriminator.train()
            self.trainer.model.source_generator.train()
            self.trainer.model.source_encoder.eval()
            # self.trainer.model.classifier.eval()
            self.trainer.experiment.log_current_epoch(e)

            self.trainer.train_data_loader.dataset.train()
            target_validator.validate(trainer)
            source_encoder_validator.validate(trainer)
