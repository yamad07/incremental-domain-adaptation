from comet_ml import Experiment, OfflineExperiment
import torch.utils.data as data
from torchvision import transforms

from src.trainers import (
        DATrainerComponent,
        SMTrainerComponent,
        CDATrainerComponent,
        CSMTrainerComponent
        )
from src.trainers.incrementals.mnist import IncrementalMnistTrainer
from src.trainers.incrementals.cityscapes import IncrementalCityscapesTrainer
from src.datasets import (
        IDAMNIST,
        IDASVHN,
        )
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
        Decoder,
        )
from src.models.incrementals.adversarial import IncrementalAdversarialModel
from src.models.incrementals.conditional_adversarial import IncrementalConditionalAdversarialModel
from src.analyzers import TargetImageSaver, TargetFeatureVisualizer


experiment = Experiment(api_key="laHAJPKUmrD2TV2dIaOWFYGkQ",
                        project_name="iada", workspace="yamad07")

source_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.25, )),
])
target_transform = transforms.Compose([
    transforms.Resize((22, 28)),
    transforms.Pad((0, 3, 0, 3)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.25, )),
])
svhn_dataset = IDASVHN(
    root='./data/',
    download=True,
    )
train_data_loader = data.DataLoader(svhn_dataset, batch_size=64, shuffle=True)

mnist_dataset = IDAMNIST(
    root='./data/',
    download=True,
    )
validate_data_loader = data.DataLoader(
    mnist_dataset, batch_size=64, shuffle=True)


model = IncrementalConditionalAdversarialModel(
    classifier=DANNClassifier(576),
    domain_discriminator=DANNDomainDiscriminator(4000),
    source_generator=CDANNSourceGenerator(z_dim=128, n_features=576, n_classes=10),
    source_discriminator=DANNSourceDiscriminator(4000),
    source_encoder=DANNEncoder(),
    target_encoder=DANNEncoder(),
    n_classes=10,
    n_features=576,
    n_random_features=4000,
)

trainer = IncrementalMnistTrainer(
        model=model,
        trainer_component_list=[
            CSMTrainerComponent(),
            CDATrainerComponent(),
        ],
        epoch_component_list=[100, 30],
        experiment=experiment,
        train_data_loader=train_data_loader,
        valid_data_loader=validate_data_loader,
        cuda_id=2,
        size_list=[0, 2, 3, 4, 5, 6, 7, 8],
        analyzer_list=[
            TargetImageSaver(),
            ]
        )
trainer.train()
