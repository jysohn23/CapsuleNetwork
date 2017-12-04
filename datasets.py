"""
All dataset classes used for training/testing of Capsule Network
"""
import os
import torch
import random
import logging
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST
from skimage import transform as skitransform

# Augmentation Configuration Constants
AUGMENT_ROT = 30
FLIP_LR_PROB = 0.5
FLIP_UP_PROB = 0.5
AUGMENT_PROB = 0.7
AUGMENT_TRANS = 10
AUGMENT_SCALE = 0.3


class FashionMNISTWrapper(MNIST):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

    def __init__(self, augment, root_value, train_bool, augment_dict=None, dir_name=None, load_file=None):
        super(FashionMNISTWrapper, self).__init__(root=root_value, train=train_bool, download=True)
        self.dir = dir_name
        self.load_txt = load_file
        self.aug_img = augment
        self.main_arr = []
        self.aug_rot = AUGMENT_ROT * np.pi/180
        self.aug_prob = AUGMENT_PROB
        self.aug_scale = AUGMENT_SCALE
        self.aug_trans = AUGMENT_TRANS
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.aug_fliplr = FLIP_LR_PROB

        if augment_dict is not None:
            self.aug_rot = np.pi/180 * augment_dict['rotation']
            self.aug_scale = augment_dict['scale']
            self.aug_trans = augment_dict['trans']
            self.aug_prob = augment_dict['prob']
            self.aug_fliplr = augment_dict['fliplr']
        if augment is True:
            logging.info('Augmentation Parameters: Rotation:{} Scale:{} Translate:{} Probablity:{} Flip_LR:{}'
                         .format(self.aug_rot, self.aug_scale, self.aug_trans, self.aug_prob, self.aug_fliplr))
            logging.info('Note: augmentation rotation is in radians')
            self.aug_lscale = 1 - self.aug_scale
            self.aug_uscale = 1 + self.aug_scale

    def augment_img(self,img):
        """
        Performs augmentation of the image with the standard parameters
        :return: img still a numpy array
        """
        if random.random() < self.aug_prob:
            rotation_angle = random.uniform(-self.aug_rot,self.aug_rot)
            scale_value = random.uniform(self.aug_lscale,self.aug_uscale)
            if random.random() < self.aug_fliplr:
                img = np.fliplr(img)
            # Doing the affine transform
            img_transform = skitransform.AffineTransform(scale=(scale_value, scale_value), rotation=rotation_angle,
                                                         translation=(random.uniform(-self.aug_trans,self.aug_trans),
                                                                      random.uniform(-self.aug_trans,self.aug_trans)))
            img = skitransform.warp(img, img_transform)
        return img

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        img = img.numpy()
        if self.aug_img is True:
            self.augment_img(img)
        return torch.from_numpy(img).unsqueeze(0).float(), target


class MNISTWrapper(MNIST):

    def __init__(self, augment, root_value, train_bool, augment_dict=None, dir_name=None, load_file=None):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,),(0.3081,))])
        super(MNISTWrapper, self).__init__(root=root_value, train=train_bool, download=True, transform=self.transform)
        self.dir = dir_name
        self.load_txt = load_file
        self.aug_img = augment
        self.main_arr = []
        self.aug_rot = AUGMENT_ROT * np.pi/180
        self.aug_prob = AUGMENT_PROB
        self.aug_scale = AUGMENT_SCALE
        self.aug_trans = AUGMENT_TRANS
        if augment_dict is not None:
            self.aug_rot = np.pi/180 * augment_dict['rotation']
            self.aug_scale = augment_dict['scale']
            self.aug_trans = augment_dict['trans']
            self.aug_prob = augment_dict['prob']
        if augment is True:
            logging.info('Augmentation Parameters: Rotation:{} Scale:{} Translate:{} Probablity:{}'
                         .format(self.aug_rot, self.aug_scale, self.aug_trans, self.aug_prob))
            logging.info('Note: augmentation rotation is in radians')
            self.aug_lscale = 1 - self.aug_scale
            self.aug_uscale = 1 + self.aug_scale

    def augment_img(self,img):
        """
        Performs augmentation of the image with the standard parameters
        :return: img still a numpy array
        """
        if random.random() < self.aug_prob:
            rotation_angle = random.uniform(-self.aug_rot,self.aug_rot)
            scale_value = random.uniform(self.aug_lscale,self.aug_uscale)
            # Doing the affine transform
            img_transform = skitransform.AffineTransform(scale=(scale_value, scale_value), rotation=rotation_angle,
                                                         translation=(random.uniform(-self.aug_trans,self.aug_trans),
                                                                      random.uniform(-self.aug_trans,self.aug_trans)))
            img = skitransform.warp(img.numpy(), img_transform)
        else:
            img = img.numpy()
        return img

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.aug_img is True:
            img = self.augment_img(img)
        else:
            img = img.numpy()

        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_dataset(dataset_name, augment=False, train_bool=True, root_value=os.getcwd()+'/'):
    if dataset_name == "MNIST":
        return MNISTWrapper(augment, root_value, train_bool)
    elif dataset_name == "FASHION_MNIST":
        return FashionMNISTWrapper(augment, root_value, train_bool)
    exit("Invalid Dataset Name")
