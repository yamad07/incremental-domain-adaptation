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
from src.trainers import (
        DATrainerComponent,
        SMTrainerComponent,
        SegCDATrainerComponent,
        SegCSMTrainerComponent
        )
from src.trainers.incrementals.mnist import IncrementalMnistTrainer
from src.trainers.incrementals.cityscapes import IncrementalCityscapesTrainer
from src.datasets.cityscapes import CityscapesDataset
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
        ResNet50Encoder,
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
train_cityscapes_dataset = CityscapesDataset(
        cityscapes_data_path='/data2/yamad/leftImg8bit_trainvaltest',
        cityscapes_meta_path='/data2/yamad/cityscapes/gtFine_trainvaltest/gtFine',
        train_city_list = ["zurich/", "weimar/", "ulm/"]
        )

train_data_loader = data.DataLoader(train_cityscapes_dataset, batch_size=16, shuffle=True)

val_cityscapes_dataset = CityscapesDataset(
        cityscapes_data_path='/data2/yamad/leftImg8bit_trainvaltest',
        cityscapes_meta_path='/data2/yamad/cityscapes/gtFine_trainvaltest/gtFine',
        train_city_list = ["zurich/"]
        )
val_data_loader = data.DataLoader(
    val_cityscapes_dataset, batch_size=16, shuffle=True)

# VGG
num_features = 32768
# model = IncrementalConditionalAdversarialModel(
#     classifier=Decoder(34, 512, 128),
#     domain_discriminator=DANNDomainDiscriminator(num_features),
#     source_generator=DANNSourceGenerator(z_dim=128, num_features=num_features),
#     source_discriminator=DANNSourceDiscriminator(num_features),
#     source_encoder=VGGEncoder(),
#     target_encoder=VGGEncoder(),
#     n_classes=8388608,
#     n_features=32768,
#     n_random_features=4000,
# )
model = IncrementalAdversarialModel(
    classifier=Decoder(34, 512, 64),
    domain_discriminator=DANNDomainDiscriminator(num_features),
    source_generator=DANNSourceGenerator(z_dim=128, num_features=num_features),
    source_discriminator=DANNSourceDiscriminator(num_features),
    source_encoder=VGGEncoder(),
    target_encoder=VGGEncoder(),
)
trainer = IncrementalCityscapesTrainer(
        model=model,
        trainer_component_list=[
            # IASVTrainerComponent(),
            # SegCSMTrainerComponent(),
            # SegCDATrainerComponent(),
            SMTrainerComponent(),
            DATrainerComponent(),
        ],
        epoch_component_list=[1, 10],
        experiment=experiment,
        train_data_loader=train_data_loader,
        valid_data_loader=val_data_loader,
        cuda_id=0,
        analyzer_list=[
            # TargetImageSaver(),
            ]
        )
trainer.train()
