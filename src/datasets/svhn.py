from torchvision.datasets import SVHN
from torchvision import transforms


class IDASVHN(SVHN):
    def __init__(
            self,
            root,
            train=True,
            download=False):
        super(
            IDASVHN,
            self).__init__(
            root=root,
            download=download)
        self.set_digit_height(0)

    def set_digit_height(self, digit_height):
        self.digit_height = digit_height
        self.transform = transforms.Compose([
                transforms.Resize((int(28 - digit_height * 2), 28)),
                transforms.Pad((0, digit_height, 0, digit_height)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, )),
            ])

    def train(self):
        pass

    def eval(self):
        pass

    @property
    def domain_name(self):
        return 'height_{}'.format(str(28 - self.digit_height * 2))
