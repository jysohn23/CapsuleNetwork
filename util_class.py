"""
Main Driver's Utilities Class
"""
import io
import torch
import logging
import torchvision
import numpy as np
from caps_layer import soft_max_nd
from torch.autograd import Variable
from torch.utils.data import DataLoader


def one_hot_encode(target, num_classes):
    return_vec = torch.zeros(target.size(0),num_classes)
    for idx in range(target.size(0)):
        return_vec[idx, target[idx]] = 1.0
    return Variable(return_vec)


class MainRun:

    def __init__(self, l_r_val, batch_size_val, tot_epoch, train_dataset, neural_network, loss_function, CUDA,
                 writer, dataset_name):
        self.l_r = l_r_val
        self.batch_size = batch_size_val
        self.epochs = tot_epoch
        self.main_model = neural_network
        self.train_dataset = train_dataset
        self.loss_fn = loss_function
        self.CUDA_val = CUDA
        self.writer = writer
        self.dataset_name = dataset_name
        logging.info("Successfully instantiated MainRun class")

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
        data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        logging.info('Loaded the training dataset')
        num_batch = len(data_loader)
        # Main Loop
        for epoch in range(self.epochs):
            logging.info('Starting Epoch {}'.format(epoch))
            for data in data_loader:
                tot_num += 1
                step = tot_num + (epoch * num_batch) - num_batch
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

                if self.writer is not None:
                    self.writer.add_scalar('train/total_loss', loss.data[0], step)
                    self.writer.add_scalar('train/margin_loss', margin_loss.data[0], step)
                    self.writer.add_scalar('train/reconstruction_loss', reconstruction_loss.data[0], step)

                if tot_num % 2 == 0:
                    logging.debug('Epoch {}, Tot_Batch_Num {}\n\tTotal_Loss {} Margin_Loss {} Recon_Loss {}'
                                  .format(epoch, tot_num, loss.data[0], margin_loss.data[0],
                                          reconstruction_loss.data[0]))
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
        res = torch.eq(ac, pred_num.cpu().data).float().mean()
        return res

    def get_accuracy_set(self, data_set):
        data_loader = DataLoader(data_set, batch_size=self.batch_size)
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

        # Reconstruct Image
        base_fname = self.dataset_name + "_e" + str(self.epochs) + "_b" + str(self.batch_size) + ".png"
        recons_img_fname = "recon_" + base_fname
        truth_img_fname = "truth_" + base_fname
        decoder = self.main_model.get_decoder()
        output = self.main_model(img)
        recon = decoder(output, label)
        recon_img = recon.view(-1, self.main_model.img_channel, self.main_model.img_width, self.main_model.img_height) # _, channel, width, height
        # Save Reconstruction and Ground Truth
        torchvision.utils.save_image(recon_img.cpu().data, recons_img_fname)
        torchvision.utils.save_image(img.cpu().data, truth_img_fname)

    def test(self, model_file,tr_ds,test_ds):
        # Setting up model
        self.main_model.load_state_dict(torch.load(model_file))
        logging.info('Loaded the model')
        self.main_model.train(mode=False)
        logging.info('Evaluating on training dataset')
        self.get_accuracy_set(data_set=tr_ds)
        logging.info('Evaluating on test dataset')
        self.get_accuracy_set(data_set=test_ds)

