import os
import torchvision.utils as vutils


class TargetImageSaver:

    def __init__(self, analyze_dir='results'):
        self.image_dir = os.path.join(analyze_dir, 'target-images')
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def analyze(self, trainer):
        self.trainer = trainer
        for source_images, source_labels in self.trainer.train_data_loader:

            digit_height = self.trainer.train_data_loader.dataset.digit_height
            image_name = str(digit_height).zfill(3)
            image_name = '{}.jpg'.format(image_name)
            image_path = os.path.join(self.image_dir, image_name)

            vutils.save_image(target_images, image_path, normalize=True)
            break
