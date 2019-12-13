from comet_ml import Experiment
from src.models.incrementals.adversarial import IncrementalAdversarialModel
from src.trainers.incrementals.adversarial import IncrementalAdversarialTrainer
from src.dataset.incrementals.mnist import IDAMNIST
import torch.utils.data as data
from torchvision import transforms


print('Start Experimentation')
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
mnist_dataset = IDAMNIST(root='./data/', download=True, source_transform=source_transform, target_transform=target_transform)
train_data_loader = data.DataLoader(mnist_dataset, batch_size=16, shuffle=True)

validate_mnist_dataset = IDAMNIST(root='./data/', train=False, download=True, source_transform=source_transform, target_transform=target_transform)
validate_data_loader = data.DataLoader(validate_mnist_dataset, batch_size=16, shuffle=True)

model = IncrementalAdversarialModel(
        classifier=Classifier(),
        domain_discriminator=DomainDiscriminator(),
        source_generator=SDMG(),
        source_discriminator=SDMD(),
        source_encoder=SourceEncoder(),
        target_encoder=TargetEncoder(),
        )

domain_adversarial_trainer = IncrementalAdversarialTrainer(
        experiment=experiment,
        model=model,
        data_loader=data_loader,
        valid_data_loader=validate_data_loader
        )
size_list = [1, 3, 5, 7]
domain_adversarial_trainer.train(3, 20)
for size in size_list:
    target_transform = transforms.Compose([
        transforms.Resize((int(28 - size * 2), 28)),
        transforms.Pad((0, size, 0, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    mnist_dataset = DAMNIST(root='./data/', download=True, source_transform=source_transform, target_transform=target_transform)
    data_loader = data.DataLoader(mnist_dataset, batch_size=16, shuffle=True)
    validate_mnist_dataset = DAMNIST(root='./data/', train=False, download=True, source_transform=source_transform, target_transform=target_transform)
    validate_data_loader = data.DataLoader(validate_mnist_dataset, batch_size=16, shuffle=True)
    domain_adversarial_trainer.set_loader(data_loader)
    domain_adversarial_trainer.val_set_loader(validate_data_loader)
    domain_adversarial_trainer.train_da(10)
    domain_adversarial_trainer.validate(validate_data_loader)
