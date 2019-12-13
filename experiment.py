from comet_ml import Experiment
from src.models.incrementals.adversarial import IncrementalAdversarialModel
from src.trainers.incrementals.adversarial import IncrementalAdversarialTrainer
from src.datasets.incrementals.mnist import IDAMNIST
from src.models.incrementals.components import Classifier, DomainDiscriminator, RandomizedMultilinear, SDMG, SDMD, SourceEncoder, TargetEncoder
import torch.utils.data as data
from torchvision import transforms


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

incremental_adversarial_trainer = IncrementalAdversarialTrainer(
    experiment=experiment,
    model=model,
    train_data_loader=train_data_loader,
    valid_data_loader=validate_data_loader,
    cuda_id=1
)
size_list = [1, 3, 5, 7]
incremental_adversarial_trainer.train(10, 20, 20)
for size in size_list:
    target_transform = transforms.Compose([
        transforms.Resize((int(28 - size * 2), 28)),
        transforms.Pad((0, size, 0, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    mnist_dataset = DAMNIST(
        root='./data/',
        download=True,
        source_transform=source_transform,
        target_transform=target_transform)
    data_loader = data.DataLoader(mnist_dataset, batch_size=16, shuffle=True)
    validate_mnist_dataset = DAMNIST(
        root='./data/',
        train=False,
        download=True,
        source_transform=source_transform,
        target_transform=target_transform)
    validate_data_loader = data.DataLoader(
        validate_mnist_dataset, batch_size=16, shuffle=True)
    incremental_adversarial_trainer.set_loader(data_loader)
    incremental_adversarial_trainer.validate_set_loader(validate_data_loader)
    incremental_adversarial_trainer.adaptation(30)
