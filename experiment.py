from comet_ml import Experiment, OfflineExperiment
import torch.utils.data as data
from torchvision import transforms

from src.trainers.incrementals.components.adversarials import (
        IADATrainerComponent,
        IASMTrainerComponent,
        IASVTrainerComponent
        )
from src.trainers.incrementals.components.conditional_adversarial import (
        ICADATrainerComponent,
        ICASMTrainerComponent,
        )
from src.trainers.incrementals.mnist import IncrementalMnistTrainer
from src.trainers.incrementals.cityscapes import IncrementalCityscapesTrainer
from src.datasets.incrementals.mnist import IDAMNIST
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
mnist_dataset = IDAMNIST(
    root='./data/',
    download=True,
    source_transform=source_transform,
    target_transform=target_transform)
train_data_loader = data.DataLoader(mnist_dataset, batch_size=16, shuffle=True)

validate_mnist_dataset = IDAMNIST(
    root='./data/',
    train=False,
    download=True,
    source_transform=source_transform,
    target_transform=target_transform)
validate_data_loader = data.DataLoader(
    validate_mnist_dataset, batch_size=16, shuffle=True)

# model = IncrementalAdversarialModel(
#     classifier=DANNClassifier(),
#     domain_discriminator=DANNDomainDiscriminator(),
#     source_generator=DANNSourceGenerator(z_dim=128),
#     source_discriminator=DANNSourceDiscriminator(),
#     source_encoder=DANNEncoder(),
#     target_encoder=DANNEncoder(),
# )
model = IncrementalConditionalAdversarialModel(
    # classifier=DANNClassifier(),
    classifier=Decoder(19),
    domain_discriminator=DANNDomainDiscriminator(4000),
    source_generator=CDANNSourceGenerator(z_dim=128, n_classes=19),
    source_discriminator=DANNSourceDiscriminator(4000),
    # source_encoder=DANNEncoder(),
    # target_encoder=DANNEncoder(),
    source_encoder=VGGEncoder(),
    target_encoder=VGGEncoder(),
    n_classes=10,
    n_features=1568,
    n_random_features=4000,
)


trainer = IncrementalCityscapesTrainer(
        model=model,
        trainer_component_list=[
            # IASVTrainerComponent(),
            ICASMTrainerComponent(),
            ICADATrainerComponent(),
        ],
        epoch_component_list=[50, 10],
        experiment=experiment,
        train_data_loader=train_data_loader,
        valid_data_loader=validate_data_loader,
        cuda_id=0,
        size_list=[6, 7, 8],
        analyzer_list=[
            TargetImageSaver(),
            ]
        )
trainer.train()
