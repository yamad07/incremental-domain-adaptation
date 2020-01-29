import torch
import torch.optim as optim
from torchvision import transforms


class IncrementalCityscapesTrainer:
    def __init__(self,
                 model,
                 trainer_component_list,
                 epoch_component_list,
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
        self.source_city_list = ["zurich/"]

        self.target_city_list = ["tubingen/", "stuttgart/", "strasbourg/", "monchengladbach/",
                "krefeld/", "hanover/", "hamburg/", "erfurt/", "dusseldorf/",
                "darmstadt/", "cologne/", "bremen/", "bochum/", "aachen/",
                "frankfurt/", "munster/", "lindau/", "berlin", "bielefeld",
                "bonn", "leverkusen", "mainz", "munich"]


        self.train_data_loader = train_data_loader
        self.validate_data_loader = valid_data_loader
        self.batch_size = train_data_loader.batch_size
        self.device = torch.device("cuda:{}".format(
            int(cuda_id)) if torch.cuda.is_available() else "cpu")
        self.analyzer_list = analyzer_list

#         self.classifier_optim = optim.Adam(
#             self.model.classifier.parameters(), lr=lr, weight_decay=5e-4)
#         self.source_optim = optim.Adam(
#             self.model.source_encoder.parameters(), lr=1e-4, weight_decay=5e-4)
#         self.target_optim = optim.Adam(
#             self.model.target_encoder.parameters(), lr=1e-4, weight_decay=5e-4)
#         self.discrim_optim = optim.Adam(
#             self.model.domain_discriminator.parameters(), lr=1e-4, weight_decay=5e-4)
#         self.source_domain_discriminator_optim = optim.Adam(
#             self.model.source_discriminator.parameters(), lr=1e-4, weight_decay=5e-4)
#         self.source_domain_generator_optim = optim.Adam(
#             self.model.source_generator.parameters(), lr=1e-4, weight_decay=5e-4)
#
        self.classifier_optim = optim.SGD(
            self.model.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.source_optim = optim.SGD(
            self.model.source_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.target_optim = optim.SGD(
            self.model.target_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.discrim_optim = optim.SGD(
            self.model.domain_discriminator.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.source_domain_discriminator_optim = optim.SGD(
            self.model.source_discriminator.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.source_domain_generator_optim = optim.SGD(
            self.model.source_generator.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        train_dataset = self.train_data_loader.dataset
        source_modeling = self.trainer_component_list[0]
        epoch = self.epoch_component_list[0]
        source_modeling.train(epoch, self)

        self.model.target_encoder.load_state_dict(self.model.source_encoder.state_dict())
        torch.save(self.model.target_encoder.state_dict(), 'weights/target_encoder.path')
        torch.save(self.model.source_generator.state_dict(), 'weights/source_generator.path')
        torch.save(self.model.source_encoder.state_dict(), 'weights/source_encoder.path')
        torch.save(self.model.classifier.state_dict(), 'weights/classifier.path')

        for city in self.target_city_list:
            train_dataset = self.train_data_loader.dataset
            train_dataset.set_cities([city])
            val_dataset = self.validate_data_loader.dataset
            val_dataset.set_cities([city])

            domain_adaptation_trainer = self.trainer_component_list[-1]
            domain_adaptation_trainer.train(self.epoch_component_list[-1], self)
