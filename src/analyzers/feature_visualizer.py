import torch
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class TargetFeatureVisualizer:

    def __init__(self, analyzer_dir='results'):
        self.image_dir = os.path.join(analyzer_dir, 'target-feature-visualizers')
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def visualize(self, trainer, e):
        self.trainer = trainer
        train_batch_size = trainer.train_data_loader.batch_size

        digit_height = trainer.train_data_loader.dataset.digit_height
        target_features_batch = torch.zeros(1000, self.trainer.model.n_features)
        source_features_batch = torch.zeros(1000, self.trainer.model.n_features)
        self.trainer.model.source_generator.eval()
        for i, (target_images, target_labels) in enumerate(trainer.train_data_loader.dataset):
            if i >= 1000:
                break
            z = torch.randn(1, trainer.model.source_generator.z_dim)
            z = z.to(self.trainer.device).detach()

            target_labels = torch.LongTensor([target_labels]).to(self.trainer.device).view(1)
            source_features = self.trainer.model.source_generator(z, target_labels)
            source_features = source_features.detach().cpu()
            source_features_batch[i, :] = source_features

            target_images = target_images.to(self.trainer.device)
            target_features = self.trainer.model.target_encoder(target_images.unsqueeze(0))
            target_features = target_features.detach().cpu()
            target_features_batch[i, :] = target_features.view(target_features.size(0), -1)

        features_batch = torch.cat([source_features_batch, target_features_batch], dim=0).numpy()
        tsne = TSNE()
        features_tsne = tsne.fit_transform(features_batch)

        plt.figure()
        plt.scatter(features_tsne[1000:, 0], features_tsne[1000:, 1], c='blue')
        plt.title("Features of Target Encoder and Source Generator")
        plt.tick_params(
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
        plt.show()

        img_name = 'target_only_height_{}_epoch_{}.png'.format(
                str(digit_height).zfill(2),
                str(e).zfill(3)
                )
        plt.savefig(os.path.join(self.image_dir, img_name))
        plt.savefig(os.path.join(self.image_dir, 'target_only_feature_visualize_last.png'))

        plt.figure()
        plt.scatter(features_tsne[:1000, 0], features_tsne[:1000, 1], c='red')
        plt.scatter(features_tsne[1000:, 0], features_tsne[1000:, 1], c='blue')
        plt.title("Features of Target Encoder and Source Generator")
        plt.tick_params(
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
        plt.show()

        img_name = 'feature_visualize_height_{}_epoch_{}.png'.format(
                str(digit_height).zfill(2),
                str(e).zfill(3)
                )
        plt.savefig(os.path.join(self.image_dir, img_name))
        plt.savefig(os.path.join(self.image_dir, 'feature_visualize_last.png'))



class GeneratedConditionalSourceFeatureVisualizer:

    def __init__(self, analyzer_dir='results'):
        self.image_dir = os.path.join(analyzer_dir, 'conditional-source-feature-visualizers')
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def visualize(self, trainer, e):
        if e % 10 == 0:
            return

        self.trainer = trainer
        train_batch_size = trainer.train_data_loader.batch_size
        source_features_list = []
        source_features_batch = torch.zeros(1000, self.trainer.model.n_features)
        trainer.model.source_generator.eval()
        for i, (source_images, source_labels) in enumerate(trainer.train_data_loader.dataset):
            if i >= 1000:
                break

            z = torch.randn(1, trainer.model.source_generator.z_dim)
            z = z.to(self.trainer.device).detach()
            source_labels = torch.LongTensor([source_labels]).to(self.trainer.device).view(1)
            source_features = self.trainer.model.source_generator(z, source_labels)
            source_features = source_features.detach().cpu()
            source_features_batch[i, :] = source_features
        source_features_batch = source_features_batch.numpy()
        tsne = TSNE()
        source_features_tsne = tsne.fit_transform(source_features_batch)

        plt.figure()
        plt.scatter(source_features_tsne[:, 0], source_features_tsne[:, 1], c='red')
        plt.title("Features of Source Generator")
        plt.tick_params(
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
        plt.show()

        img_name = 'feature_visualize_{}.png'.format(str(e).zfill(3))
        plt.savefig(os.path.join(self.image_dir, img_name))
        plt.savefig(os.path.join(self.image_dir, 'feature_visualize_last.png'))

class GeneratedSourceFeatureVisualizer:

    def __init__(self, analyzer_dir='results'):
        self.image_dir = os.path.join(analyzer_dir, 'source-feature-visualizers')
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def visualize(self, trainer, e):
        if e % 10 == 0:
            return

        self.trainer = trainer
        train_batch_size = trainer.train_data_loader.batch_size
        source_features_list = []
        source_features_batch = torch.zeros(1000, self.trainer.model.n_features)
        self.trainer.model.source_generator.eval()
        for i, (source_images, source_labels) in enumerate(trainer.train_data_loader.dataset):
            if i >= 1000:
                break

            z = torch.randn(1, trainer.model.source_generator.z_dim)
            z = z.to(self.trainer.device).detach()
            source_features = self.trainer.model.source_generator(z)
            source_features = source_features.detach().cpu().view(source_features.size(0), -1)
            source_features_batch[i, :] = source_features
        source_features_batch = source_features_batch.numpy()
        tsne = TSNE()
        source_features_tsne = tsne.fit_transform(source_features_batch)

        plt.figure()
        plt.scatter(source_features_tsne[:, 0], source_features_tsne[:, 1], c='red')
        plt.title("Features of Source Generator")
        plt.tick_params(
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
        plt.show()

        img_name = 'feature_visualize_{}.png'.format(str(e).zfill(3))
        plt.savefig(os.path.join(self.image_dir, img_name))
        plt.savefig(os.path.join(self.image_dir, 'feature_visualize_last.png'))

