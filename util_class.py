# Main utility functions
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import transform as skitransform
import random
import torch
from torch.autograd import Variable
import io
import logging
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from caps_layer import soft_max_nd
import torchvision

def one_hot_encode(target,num_classes):
    return_vec = torch.zeros(target.size(0),num_classes)
    for idx in range(target.size(0)):
        return_vec[idx,target[idx]] = 1
    return Variable(return_vec)

class master_dataset(MNIST):
    def __init__(self,augment,root_value,train_bool,augment_dict=None,dir_name=None,load_file=None):
        super(master_dataset, self).__init__(root=root_value,train=train_bool,download=True)
        self.dir = dir_name
        self.load_txt = load_file
        self.aug_img = augment
        self.main_arr = []
        self.aug_rot =30 * np.pi/180
        self.aug_prob = 0.7
        self.aug_scale = 0.3
        self.aug_trans = 10
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.aug_fliplr = 0.5
        # self.aug_flipud = 0.5
        # self.img_size = img_size
        # Getting the pairs from the given file and loading into array
        # fid = io.open(self.dir + "/" + load_file, 'r', encoding='utf-8')
        # for line in fid.readlines():
        #     self.main_arr.append(line.split())
        # fid.close()
        if augment_dict is not None:
            self.aug_rot = np.pi/180 * augment_dict['rotation']
            self.aug_scale = augment_dict['scale']
            self.aug_trans = augment_dict['trans']
            self.aug_prob = augment_dict['prob']
            # self.aug_fliplr = augment_dict['fliplr']
            # self.aug_flipud = augment_dict['flipud']
        if augment is True:
            logging.info('Augmentation Parameters: Rotation:{} Scale:{} Translate:{} Probablity:{}'.format(self.aug_rot,self.aug_scale,self.aug_trans,self.aug_prob))#,self.aug_flipud,self.aug_fliplr))
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
            # if random.random() < self.aug_flipud:
            #     img = np.flipud(img)
            # if random.random() < self.aug_fliplr:
            #     img = np.fliplr(img)
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

class main_run:
    def __init__(self, l_r_val, batch_size_val, tot_epoch, train_loader,neural_network,loss_function,CUDA):
        self.l_r = l_r_val
        self.batch_size = batch_size_val
        self.epochs = tot_epoch
        self.main_model = neural_network
        self.train_data_loader = train_loader
        self.loss_fn = loss_function
        self.CUDA_val = CUDA

    def train(self, model_file, load_param=None):
        logging.info('Learning Rate Is: {} Batch Size: {} Epochs: {}'.format(self.l_r, self.batch_size, self.epochs))
        if load_param is not None:
            logging.info('Loading Parameters In Model From File: {}'.format(load_param))
            self.main_model.load_state_dict(torch.load(load_param))
        # Setting up optimizer
        optimizer = torch.optim.Adam(self.main_model.parameters(), lr=self.l_r)
        # File for printing the loss function
        file_id = io.open('loss_track.txt', 'ab')

        tot_num = 0
        data_loader = DataLoader(self.train_data_loader, batch_size=self.batch_size)
        logging.info('Loaded the training dataset')

        # Main Loop
        for epoch in range(self.epochs):
            logging.info('Starting Iteration {}'.format(epoch))
            for data in data_loader:
                # Finding the predicted label and getting the loss function
                img, label = data
                if self.CUDA_val is True:
                    img, label = Variable(img).cuda(), one_hot_encode(target=label,num_classes=10).cuda()
                else:
                    img, label = Variable(img), one_hot_encode(target=label,num_classes=10)
                # zero the gradients
                optimizer.zero_grad()
                # Get label
                predicted_label = self.main_model(img)
                # Getting the loss
                loss, margin_loss, reconstruction_loss = self.loss_fn(img, predicted_label, label)
                # Back propogating the loss
                loss.backward(retain_graph=True)
                # Using optimizer
                optimizer.step()
                tot_num += 1
                if tot_num % 2 == 0:
                    logging.debug('Epoch {}, Tot_Num {} Loss {}'.format(epoch, tot_num, loss.data[0]))
                    curr_str = '{} \n'.format(loss.data[0])
                    file_id.writelines(curr_str)
            logging.debug('Epoch {}, Tot_Num {} Loss {}'.format(epoch, tot_num, loss.data[0]))
        file_id.close()
        # Save the parameters
        torch.save(self.main_model.state_dict(), model_file)

    def accuracy_func(self, pred, ac):
        pred_len = torch.sqrt((pred ** 2).sum(dim=2, keepdim=True))
        soft_max_return = soft_max_nd(pred_len, 1)
        _, max_idx = soft_max_return.max(dim=1)
        pred_num = max_idx.squeeze()
        junk = torch.eq(ac, pred_num.cpu().data).float().mean()
        # print(junk)
        return junk

    def get_accuracy_set(self, data_set):
        data_loader = DataLoader(data_set, batch_size=100)
        main_arr = np.array([])
        counter = 0
        for data in data_loader:
            # Finding the predicted label and getting the loss function
            img, label = data
            if self.CUDA_val is True:
                img = Variable(img).cuda()
            else:
                img = Variable(img)
            main_arr = np.concatenate((main_arr, np.array([self.accuracy_func(self.main_model(img), label)])))
            counter += 1
            if counter%2 == 0:
                logging.debug('Current Accuracy: {}'.format(np.mean(main_arr)))
        logging.info('Total Samples: {} Accuracy: {}'.format(data_set.__len__(), np.mean(main_arr)))

    def test(self, model_file,tr_ds,test_ds):
        # Setting up model
        self.main_model.load_state_dict(torch.load(model_file))
        logging.info('Loaded the model')
        self.main_model.train(mode=False)
        logging.info('Evaluating on training dataset')
        self.get_accuracy_set(data_set=tr_ds)
        logging.info('Evaluating on test dataset')
        self.get_accuracy_set(data_set=test_ds)