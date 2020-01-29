import torch
import torch.nn.functional as F
import numpy as np
from ..analyzers import (
        TargetFeatureVisualizer,
        SourceEncoderAccuracyValidator,
        TargetEncoderAccuracyValidator,
        SourceGeneratorAccuracyValidator,
        TargetKLDValidator,
        SourceKLDValidator
        )
from .losses.mmd import RBFMMDLoss


class MMDTrainerComponent:

    def train(self, epoch, trainer):
        self.trainer = trainer
        self.train_batch_size = self.trainer.train_data_loader.batch_size
        self.valid_batch_size = self.trainer.validate_data_loader.batch_size
        target_validator = TargetEncoderAccuracyValidator()
        source_encoder_validator = SourceEncoderAccuracyValidator()
        source_generator_validator = SourceEncoderAccuracyValidator()
        target_kld_validator = TargetKLDValidator()
        source_kld_validator = SourceKLDValidator()
        self.mmd_criterion = RBFMMDLoss()

        for e in range(epoch):
            self.trainer.model.target_encoder.train()
            self.trainer.model.source_encoder.train()
            self.trainer.model.source_generator.train()
            self.trainer.experiment.log_current_epoch(e)

            self.trainer.train_data_loader.dataset.train()
            for i, (data, labels) in enumerate(self.trainer.train_data_loader):
                data = data.to(self.trainer.device)
                labels = labels.to(self.trainer.device)

                mmd_loss = self._ad_train_target_encoder(data)

                self.trainer.experiment.log_metric('mmd_loss', mmd_loss.item())

            target_validator.validate(trainer)
            source_encoder_validator.validate(trainer)
            source_generator_validator.validate(trainer)
            target_kld_validator.validate(trainer)
            source_kld_validator.validate(trainer)

            print("Epoch: {0} MMD: {1}".format( e, mmd_loss.item()))

    def _ad_train_target_encoder(self, target_data):
        batch_size = target_data.size(0)
        # init
        self.trainer.target_optim.zero_grad()
        self.trainer.source_optim.zero_grad()
        self.trainer.discrim_optim.zero_grad()

        # forward
        target_features = self.trainer.model.target_encoder(target_data)

        z = torch.randn(self.train_batch_size, self.trainer.model.source_generator.z_dim).to(self.trainer.device)
        source_features = self.trainer.model.source_generator(z)

        mmd_loss = self.mmd_criterion(target_features, source_features)

        # backward
        mmd_loss.backward()
        self.trainer.target_optim.step()
        return mmd_loss
