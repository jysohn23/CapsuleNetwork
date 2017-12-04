"""
Main Driver's Utilities Class
"""
import io
import torch
import logging
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader


def one_hot_encode(target, num_classes):
    return_vec = torch.zeros(target.size(0),num_classes)
    for idx in range(target.size(0)):
        return_vec[idx, target[idx]] = 1.0
    return Variable(return_vec)


class MainRun:

    def __init__(self, l_r_val, batch_size_val, tot_epoch, train_loader, neural_network, loss_function, CUDA):
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
            logging.info('Starting Epoch {}'.format(epoch))
            for data in data_loader:
                # Finding the predicted label and getting the loss function
                img, label = data
                if self.CUDA_val is True:
                    img, label = Variable(img).cuda(), one_hot_encode(target=label, num_classes=10).cuda()
                else:
                    img, label = Variable(img), one_hot_encode(target=label, num_classes=10)
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

                # DEBUGGING
                logging.debug('total_loss: {}; margin_loss: {}; recon_loss: {}'
                              .format(loss, margin_loss, reconstruction_loss))

                tot_num += 1
                if tot_num % 2 == 0:
                    logging.debug('Epoch {}, Tot_Batch_Num {} Loss {}'.format(epoch, tot_num, loss.data[0]))
                    curr_str = '{} \n'.format(loss.data[0])
                    file_id.writelines(curr_str)

        file_id.close()
        # Save the model
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

