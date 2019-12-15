from comet_ml import Experiment
import torch.utils.data as data
from torchvision import transforms

from src.trainers.incrementals.components.adversarial import IncrementalAdversarialTrainer
from src.trainers.incrementals.mnist import IncrementalMnistTrainer
from src.datasets.incrementals.mnist import IDAMNIST
from src.models.incrementals.components import (
        Classifier,
        DomainDiscriminator,
        RandomizedMultilinear,
        SDMG,
        SDMD,
        SourceEncoder,
        TargetEncoder
        )
from src.models.incrementals.adversarial import IncrementalAdversarialModel


experiment = Experiment(api_key="laHAJPKUmrD2TV2dIaOWFYGkQ",
                        project_name="iada", workspace="yamad07")

source_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])
target_transform = transforms.Compose([
    transforms.Resize((14, 28)),
    transforms.Pad((0, 7, 0, 7)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
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

model = IncrementalAdversarialModel(
    classifier=Classifier(),
    domain_discriminator=DomainDiscriminator(),
    source_generator=SDMG(),
    source_discriminator=SDMD(),
    source_encoder=SourceEncoder(),
    target_encoder=TargetEncoder(),
)

incremental_adversarial_trainer_component = IncrementalAdversarialTrainer(
    experiment=experiment,
    model=model,
    train_data_loader=train_data_loader,
    valid_data_loader=validate_data_loader,
    cuda_id=1
)

trainer = IncrementalMnistTrainer(
        incremental_trainer_component=incremental_adversarial_trainer_component,
        size_list=[1, 3, 5, 7],
        batch_size=16,
        )

trainer.train(1, 1, 1)
