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
        CSMTrainerComponent,
        MMDTrainerComponent
        )
from src.trainers.incrementals.components.conditional_adversarial import (
        ICADATrainerComponent,
        ICASMTrainerComponent,
        )
from src.trainers.incrementals.mnist import IncrementalMnistTrainer
from src.trainers.incrementals.cityscapes import IncrementalCityscapesTrainer
from src.datasets import (
        IDAMNIST,
        IDASVHN
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

model = IncrementalAdversarialModel(
    classifier=DANNClassifier(576),
    domain_discriminator=DANNDomainDiscriminator(576),
    source_generator=DANNSourceGenerator(z_dim=128, num_features=576),
    source_discriminator=DANNSourceDiscriminator(576),
    source_encoder=DANNEncoder(),
    target_encoder=DANNEncoder(),
    n_features=576
    )

trainer = IncrementalMnistTrainer(
        model=model,
        trainer_component_list=[
            SMTrainerComponent(),
            MMDTrainerComponent(),
        ],
        epoch_component_list=[100, 30],
        experiment=experiment,
        train_data_loader=train_data_loader,
        valid_data_loader=validate_data_loader,
        cuda_id=0,
        size_list=[0, 2, 3, 4, 5, 6, 7, 8],
        analyzer_list=[
            TargetImageSaver(),
            ]
        )
trainer.train()
