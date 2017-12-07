# Main Function
import os
import logging
from util_class import MainRun
from caps_loss import CapsuleLoss
from caps_net import CapsuleNetwork
from dataset_util import master_base,FashionMNIST
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torchvision import transforms
import torchvision
import numpy as np
from CIFAR10_net import CIFAR10nn

ds_dict = {'MNIST':1, 'FashionMNIST':2, 'CIFAR10':3, 'CIFAR100':4}

def get_ds_class(dataset,tr_ds,d_ds,aug):
    if dataset == 1:
        target_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return master_base(dataset=torchvision.datasets.MNIST(root=os.getcwd() + '/', train=tr_ds, download=d_ds),
                           gray=True,doAUG=aug,desired_transform=target_transform,flip_lr_bool=False)
    elif dataset == 2:
        target_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return master_base(dataset=FashionMNIST(root=os.getcwd() + '/', train=tr_ds, download=d_ds),
                           gray=True,doAUG=aug,desired_transform=target_transform,flip_lr_bool=False)
    elif dataset == 3:
        # TODO: change the target transform
        target_transform = transforms.Compose([transforms.ToTensor()])
        return master_base(dataset=torchvision.datasets.CIFAR10(root=os.getcwd() + '/', train=tr_ds, download=d_ds),
                           gray=False,doAUG=aug,desired_transform=target_transform,flip_lr_bool=True)
    elif dataset == 4:
        # TODO: change the target transform
        target_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return master_base(dataset=torchvision.datasets.CIFAR100(root=os.getcwd() + '/', train=tr_ds, download=d_ds),
                           gray=False,doAUG=aug,desired_transform=target_transform,flip_lr_bool=True)

def test_pics(dataloader,isGray,tot_count):
    import cv2 as cv
    """
    Outputs image to show the current image and test parameters
    """
    for data in dataloader:
        img,label = data
        logging.debug('Max from float tensor is: {}'.format(np.max(np.max(img.numpy()))))
        img = np.uint8(img.squeeze(0).numpy()*255)
        if isGray:
            img = np.transpose(img,axes = (2,1,0))
        else:
            img = np.transpose(img,axes=(1,2,0))
        cv.imshow('Sample',img)
        cv.waitKey()
        tot_count -= 1
        if tot_count == 0:
            return

def main():
    # Main Parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='')
    parser.add_argument("--load", type=str, default=None, help='File to load with the weights')
    parser.add_argument("--save", type=str, default=None, help='File to save the weights')
    parser.add_argument("-a", "--augment", default=False, action='store_true', help='Augment training data')
    parser.add_argument("--log", type=str, default='DEBUG', help='Current Logging Level')
    parser.add_argument("-e", type=int, default=10, help='Number of epochs')
    parser.add_argument("-b", type=int, default=64, help='Number of image pairs in batch')
    parser.add_argument("-l", type=float, default=0.01, help='Leaning Rate')
    parser.add_argument("-c", default=False, action='store_true', help='CUDA')
    parser.add_argument("--dataset", type=str, required=True, help='Dataset to train on. Options:\n' +
                        '1. MNIST\n2. FashionMNIST')
    parser.add_argument("-tb", "--tensorboard", default=False, action="store_true",
                        help="Save info for Tensorboard visualization")
    args = parser.parse_args()

    # Setting up logger
    logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s',level=args.log)
    if ds_dict[args.dataset] < 3:
        nn_network = CapsuleNetwork(img_channel=1, img_height=28,
                                    img_width=28, num_conv_input_channels=1, num_conv_output_channels=256,
                                    num_prim_units=8, prim_unit_size=1152, num_classes=10, output_unit_size=16,
                                    num_routing=3,
                                    CUDA=args.c, conv_kernel_size=9, prim_kernel_size=9, prim_output_channels=32)
    elif ds_dict[args.dataset] == 3:
        nn_network = CIFAR10nn(CUDA=args.c)
    # Getting the loss function
    loss_fn = CapsuleLoss(regularization_scale=0.0005,CUDA=args.c,decoder=nn_network.get_decoder())
    if args.c is True:
        loss_fn = loss_fn.cuda()
        nn_network = nn_network.cuda()

    if args.tensorboard is True:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()
    else:
        writer = None
    tr_dataset = get_ds_class(dataset=ds_dict[args.dataset], tr_ds=True, d_ds=True, aug=args.augment)
    # Main class to use
    main_class = MainRun(l_r_val=args.l,batch_size_val=args.b,tot_epoch=args.e,
                         train_dataset=tr_dataset,neural_network=nn_network,loss_function=loss_fn,CUDA=args.c,
                         writer=writer, dataset_name=args.dataset)

    # Runs the training functionality
    if args.save is not None:
        main_class.train(model_file=args.save,load_param=args.load)

    # Runs the testing functionality
    if args.load is not None:
        test_ds = get_ds_class(dataset=ds_dict[args.dataset], tr_ds=False, d_ds=True, aug=False)
        main_class.test(model_file=args.load,tr_ds=tr_dataset,test_ds=test_ds)

    if args.tensorboard is True:
        writer.close()

    # Sample to test dataset
    # from torch.utils.data import DataLoader
    # test_pics(DataLoader(tr_dataset), tr_dataset.grayscale, 4)


if __name__ == '__main__':
    main()
