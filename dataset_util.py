import random
import logging
import numpy as np
import torchvision.datasets
from skimage import transform as skitransform

# Augmentation Configuration Constants
aug_dict_pre = {'rot':0.5235987756, 'prob':0.7,'flip_lr':0.5,'trans':3,'scale':0.3}

# Making new class to support augmentation

class augment_base:
    """
    Class for image augmentation
    """
    def __init__(self,augment,transform_custom, augment_dict,flip_lr):
        self.transform = transform_custom
        self.aug_img = augment
        self.aug_dict = augment_dict
        if flip_lr is False:
            self.aug_dict['flip_lr'] = -1
        if augment is True:
            logging.info('Augmentation Parameters: Rotation:{} Scale:{} Translate:{} Probablity:{} Flip_LR:{}'
                         .format(self.aug_dict['rot'], self.aug_dict['scale'], self.aug_dict['trans'],
                                 self.aug_dict['prob'], self.aug_dict['flip_lr']))
            logging.info('Note: augmentation rotation is in radians')
            self.aug_dict['lscale'] = 1 - self.aug_dict['scale']
            self.aug_dict['uscale'] = 1 + self.aug_dict['scale']

    def augment_img(self,img):
        """
        Performs augmentation of the image if required and returns a tensor
        """
        # logging.debug('Shape of the array: {}'.format(img.shape))
        if self.aug_img is False:
            # logging.debug('No augmentation max before image transform: {}'.format(np.max(np.max(img))))
            return self.transform(img)
        if random.random() < self.aug_dict['prob']:
            img_transform = skitransform.AffineTransform(
                        scale=tuple([random.uniform(self.aug_dict['lscale'],self.aug_dict['uscale'])]*2),
                        rotation=random.uniform(-self.aug_dict['rot'],self.aug_dict['rot']),
                        translation=(random.uniform(-self.aug_dict['trans'],self.aug_dict['trans']),
                                    random.uniform(-self.aug_dict['trans'],self.aug_dict['trans'])))
            img = skitransform.warp(img, img_transform)
            if random.random() < self.aug_dict['flip_lr']:
                img = np.fliplr(img)
            # logging.debug('Augmentation max before image transform: {}'.format(np.max(np.max(img))))
            return self.transform(np.uint8(img*255))

        else:
            # logging.debug('Augmentation max before image transform (no aug): {}'.format(np.max(np.max(img))))
            return self.transform(img)


class FashionMNIST(torchvision.datasets.MNIST):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'rawFashion'
    processed_folder = 'processedFashion'
    training_file = 'training.pt'
    test_file = 'test.pt'


class master_base:
    def __init__(self,desired_transform,dataset,gray,doAUG,flip_lr_bool,aug_dict=aug_dict_pre):
        self.augmentation = augment_base(augment=doAUG,augment_dict=aug_dict,transform_custom=desired_transform,flip_lr=flip_lr_bool)
        self.main_dataset = dataset
        self.grayscale = gray
        logging.info('Successfully instantiated master base class')

    def __len__(self):
        return self.main_dataset.__len__()

    def __getitem__(self, idx):
        img,target = self.main_dataset.__getitem__(idx)
        if self.grayscale:
            img = self.augmentation.augment_img(np.expand_dims(np.array(img),axis=2))
        else:
            img = self.augmentation.augment_img(np.array(img))
        return img, target
