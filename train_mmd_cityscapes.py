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
        VGGSourceGenerator,
        Decoder,
        )
from src.models.incrementals.adversarial import IncrementalAdversarialModel
from src.models.incrementals.conditional_adversarial import IncrementalConditionalAdversarialModel
from src.analyzers import TargetImageSaver, TargetFeatureVisualizer

config = {
        "n_decoder_hidden": 128,
        "batch_size": 32
    }


experiment = Experiment(api_key="laHAJPKUmrD2TV2dIaOWFYGkQ",
                        project_name="iada", workspace="yamad07")

train_cityscapes_dataset = CityscapesDataset(
        cityscapes_data_path='/data/ubuntu/cityscapes/leftImg8bit',
        cityscapes_meta_path='/data/ubuntu/cityscapes/gtFine',
        train_city_list = ["zurich/", "weimar/", "ulm/"]
        )

train_data_loader = data.DataLoader(train_cityscapes_dataset, batch_size=config['batch_size'], shuffle=True)

val_cityscapes_dataset = CityscapesDataset(
        cityscapes_data_path='/data/ubuntu/cityscapes/leftImg8bit',
        cityscapes_meta_path='/data/ubuntu/cityscapes/gtFine',
        train_city_list = ["zurich/", "weimar/", "ulm/"]
        )
val_data_loader = data.DataLoader(
    val_cityscapes_dataset, batch_size=config['batch_size'], shuffle=True)

num_features = 32768
model = IncrementalAdversarialModel(
    classifier=Decoder(34, 512, config['n_decoder_hidden']),
    domain_discriminator=DANNDomainDiscriminator(num_features),
    source_generator=VGGSourceGenerator(z_dim=128, num_features=num_features),
    source_discriminator=DANNSourceDiscriminator(num_features),
    source_encoder=VGGEncoder(),
    target_encoder=VGGEncoder(),
    n_features=num_features
    )

trainer = IncrementalCityscapesTrainer(
        model=model,
        trainer_component_list=[
            SMTrainerComponent(),
            MMDTrainerComponent(),
        ],
        epoch_component_list=[70, 30],
        experiment=experiment,
        train_data_loader=train_data_loader,
        valid_data_loader=val_data_loader,
        cuda_id=3,
        analyzer_list=[
            # TargetImageSaver(),
            ]
        )
trainer.train()
