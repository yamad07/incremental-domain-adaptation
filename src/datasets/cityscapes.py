import torch
from torch.utils.data import Dataset

import numpy as np
import cv2
import os

dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
        "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
        "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
        "bremen/", "bochum/", "aachen/", "berlin", "bielefeld", "bonn",
        "leverkusen", "mainz", "munich", "frankfurt/", "munster/", "lindau/"]

class CityscapesDataset(Dataset):
    def __init__(
            self,
            cityscapes_data_path,
            cityscapes_meta_path,
            train_city_list=["jena/"],
            train=True,
            ):
        self.img_dir = os.path.join(cityscapes_data_path, "leftImg8bit", "train")
        self.label_dir = os.path.join(cityscapes_meta_path, "train")

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.set_cities(train_city_list)
        self.__train = train

    def set_cities(self, train_city_list):
        self.train_city_list = train_city_list
        self.examples = []
        dirs = self.train_city_list
        for train_dir in dirs:

            train_img_dir_path = os.path.join(self.img_dir, train_dir)

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = os.path.join(train_img_dir_path, file_name)

                label_img_path = os.path.join(self.label_dir, train_dir, img_id + "_gtFine_labelIds.png")

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024))

        # flip the img and the label with 0.5 probability:
        if self.__train:
            flip = np.random.randint(low=0, high=2)
            if flip == 1:
                img = cv2.flip(img, 1)
                label_img = cv2.flip(label_img, 1)

        ########################################################################
        # randomly scale the img and the label:
        ########################################################################
        if self.__train:
            scale = np.random.uniform(low=0.7, high=2.0)
        else:
            scale = 1
        new_img_h = int(scale*self.new_img_h)
        new_img_w = int(scale*self.new_img_w)

        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (new_img_w, new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w, 3))

        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (new_img_w, new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w))
        ########################################################################

        # # # # # # # # debug visualization START
        # print (scale)
        # print (new_img_h)
        # print (new_img_w)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        ########################################################################
        # select a 256x256 random crop from the img and label:
        ########################################################################
        if self.__train:
            start_x = np.random.randint(low=0, high=(new_img_w - 256))
            start_y = np.random.randint(low=0, high=(new_img_h - 256))
        else:
            start_x = new_img_w - 256
            start_y = new_img_h - 256

        end_x = start_x + 256
        end_y = start_y + 256

        img = img[start_y:end_y, start_x:end_x] # (shape: (256, 256, 3))
        label_img = label_img[start_y:end_y, start_x:end_x] # (shape: (256, 256))
        ########################################################################

        # # # # # # # # debug visualization START
        # print (img.shape)
        # print (label_img.shape)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (256, 256, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 256, 256))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 256, 256))
        label_img = torch.from_numpy(label_img).long() # (shape: (256, 256))

        return (img, label_img)

    def __len__(self):
        return len(self.examples)

    def train(self):
        self.__train = True

    def eval(self):
        self.__train = False

    @property
    def domain_name(self):
        return 'city_{}'.format(self.train_city_list[0])
