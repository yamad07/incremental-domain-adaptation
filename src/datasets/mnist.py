from torchvision.datasets.mnist import MNIST
from torchvision import transforms
import random
from PIL import Image
import torch


class IDAMNIST(MNIST):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(
            self,
            root,
            train=True,
            download=False):
        super(
            IDAMNIST,
            self).__init__(
            root=root,
            train=train,
            download=download)
        del self.train
        self.set_digit_height(0)
        self.__train = train

    def set_digit_height(self, digit_height):
        self.digit_height = digit_height
        self.transform = transforms.Compose([
                transforms.Resize((int(28 - digit_height * 2), 28)),
                transforms.Pad((0, digit_height, 0, digit_height)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, )),
            ])

    def __getitem__(self, idx):
        img = self.train_data[idx]
        label = self.train_labels[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.transform(img)
        img = torch.cat([img, img, img], dim=0)

        return img, label

    def train(self):
        self.__train = True

    def eval(self):
        self.__train = False

    def __len__(self):
        if self.__train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    @property
    def domain_name(self):
        return 'height_{}'.format(str(28 - self.digit_height * 2))
