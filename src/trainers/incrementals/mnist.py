from torchvision import transforms


class IncrementalMnistTrainer:

    def __init__(
            self,
            incremental_trainer_component,
            size_list,
            batch_size,
            ):

        self.incremental_trainer_component = incremental_trainer_component
        self.batch_size = batch_size
        self.size_list = size_list

    def train(self, sup_epoch, sdm_epoch, ida_epoch):
        self.incremental_trainer_component.train(sup_epoch, sdm_epoch, ida_epoch)
        for size in size_list:
            target_transform = transforms.Compose([
                transforms.Resize((int(28 - size * 2), 28)),
                transforms.Pad((0, size, 0, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, )),
            ])
            train_dataset = self.incremental_adversarial_trainer.train_data_loader.dataset
            train_dataset.set_target_transform(target_transform)
            val_dataset = self.incremental_adversarial_trainer.validate_data_loader.dataset
            val_dataset.set_target_transform(target_transform)
            incremental_adversarial_trainer.adaptation(30)
