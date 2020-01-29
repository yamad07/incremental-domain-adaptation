import torch
import torch.optim as optim
from torchvision import transforms


class IncrementalMnistTrainer:
    def __init__(self,
                 model,
                 trainer_component_list,
                 epoch_component_list,
                 size_list,
                 experiment,
                 train_data_loader,
                 valid_data_loader,
                 cuda_id,
                 lr=1e-3,
                 analyzer_list=[],
                 ):
        self.model = model
        self.experiment = experiment
        self.trainer_component_list = trainer_component_list
        self.epoch_component_list = epoch_component_list
        self.size_list = size_list

        self.train_data_loader = train_data_loader
        self.validate_data_loader = valid_data_loader
        self.batch_size = train_data_loader.batch_size
        self.device = torch.device("cuda:{}".format(
            int(cuda_id)) if torch.cuda.is_available() else "cpu")
        self.analyzer_list = analyzer_list

        # self.classifier_optim = optim.Adam(
        #     self.model.classifier.parameters(), lr=1e-4, weight_decay=5e-4)
        # self.source_optim = optim.Adam(
        #     self.model.source_encoder.parameters(), lr=1e-4, weight_decay=5e-4)
        # self.target_optim = optim.Adam(
        #     self.model.target_encoder.parameters(), lr=1e-4, weight_decay=5e-4)
        # self.discrim_optim = optim.Adam(
        #     self.model.domain_discriminator.parameters(), lr=1e-4, weight_decay=5e-4)
        # self.source_domain_discriminator_optim = optim.Adam(
        #     self.model.source_discriminator.parameters(), lr=1e-4, weight_decay=5e-4)
        # self.source_domain_generator_optim = optim.Adam(
        #     self.model.source_generator.parameters(), lr=1e-4, weight_decay=5e-4)

        self.classifier_optim = optim.SGD(
            self.model.classifier.parameters(), momentum=0.9, lr=lr, weight_decay=5e-4)
        self.source_optim = optim.SGD(
            self.model.source_encoder.parameters(), momentum=0.9, lr=lr, weight_decay=5e-4)
        self.target_optim = optim.SGD(
            self.model.target_encoder.parameters(), momentum=0.9, lr=lr, weight_decay=5e-4)
        self.discrim_optim = optim.SGD(
            self.model.domain_discriminator.parameters(), momentum=0.9, lr=lr, weight_decay=5e-4)
        self.source_domain_discriminator_optim = optim.SGD(
            self.model.source_discriminator.parameters(), momentum=0.9, lr=lr, weight_decay=5e-4)
        self.source_domain_generator_optim = optim.SGD(
            self.model.source_generator.parameters(), momentum=0.9, lr=lr, weight_decay=5e-4)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        train_dataset = self.train_data_loader.dataset
        source_modeling = self.trainer_component_list[0]
        epoch = self.epoch_component_list[0]
        source_modeling.train(epoch, self)
        self.model.target_encoder.load_state_dict(self.model.source_encoder.state_dict())
        for size in self.size_list:
            train_dataset = self.train_data_loader.dataset
            train_dataset.set_digit_height(size)
            val_dataset = self.validate_data_loader.dataset
            val_dataset.set_digit_height(size)

            domain_adaptation_trainer = self.trainer_component_list[1]
            domain_adaptation_trainer.train(self.epoch_component_list[1], self)
