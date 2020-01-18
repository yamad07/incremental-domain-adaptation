class DiscrepancyAnalyzer:

    def __init__(self, analyze_dir='results'):
        pass

    def analyzer(self, trainer):

        self.trainer = trainer

        for target_images target_labels in self.trainer.valid_data_loader:

            target_feature = self.trainer.target_encoder(target_images)
            self.trainer.source_generator(source_images)
