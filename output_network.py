from comet_ml import Experiment, OfflineExperiment
import torch.utils.data as data
from torchvision import transforms

from src.trainers.incrementals.components.adversarials import (
        IADATrainerComponent,
        IASMTrainerComponent,
        IASVTrainerComponent
        )
from src.trainers import (
        DATrainerComponent,
        SMTrainerComponent,
        CDATrainerComponent,
        CSMTrainerComponent
        )
from src.trainers.incrementals.components.conditional_adversarial import (
        ICADATrainerComponent,
        ICASMTrainerComponent,
        )
from src.trainers.incrementals.mnist import IncrementalMnistTrainer
from src.trainers.incrementals.cityscapes import IncrementalCityscapesTrainer
from src.datasets import IDAMNIST
from src.models.incrementals.components import (
        DANNClassifier,
        DANNEncoder,
        DANNSourceGenerator,
        CDANNSourceGenerator,
        DANNSourceDiscriminator,
        DANNDomainDiscriminator,
        Classifier,
        DomainDiscriminator,
        SourceGenerator,
        SourceDiscriminator,
        Encoder,
        VGGEncoder,
        VGGSourceGenerator,
        ResNet50Encoder,
        Decoder,
        )

num_features = 32768
classifier=Decoder(34, 512, 256),
domain_discriminator=DANNDomainDiscriminator(num_features),
source_generator=VGGSourceGenerator(z_dim=128, num_features=num_features),
source_discriminator=DANNSourceDiscriminator(num_features),
source_encoder=VGGEncoder(),
target_encoder=VGGEncoder(),
print(classifier)
print(domain_discriminator)
print(source_generator)
print(source_discriminator)
print(source_encoder)
