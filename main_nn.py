# Main Function
import os
import logging
from util_class import MainRun
from caps_loss import CapsuleLoss
from caps_net import CapsuleNetwork
from datasets import MNISTWrapper, get_dataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


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
    parser.add_argument("-d", "--dataset", type=str, required=True, help='Dataset to train on. Options:\n' +
                        '1. MNIST\n2. FASHION_MNIST')
    args = parser.parse_args()

    # Setting up logger
    logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s',level=args.log)
    tr_dataset = get_dataset(args.dataset, augment=args.augment, train_bool=True, root_value=os.getcwd()+'/')
    nn_network = CapsuleNetwork(img_channel=1,img_height=28,
                            img_width=28,num_conv_input_channels=1,num_conv_output_channels=256,
                            num_prim_units=8,prim_unit_size=1152,num_classes=10,output_unit_size=16,num_routing=3,
                            CUDA=args.c,conv_kernel_size=9,prim_kernel_size=9,prim_output_channels=32)
    loss_fn = CapsuleLoss(regularization_scale=0.0005,CUDA=args.c,decoder=nn_network.get_decoder())
    if args.c is True:
        loss_fn = loss_fn.cuda()
        nn_network = nn_network.cuda()
    # Main class to use
    main_class = MainRun(l_r_val=args.l,batch_size_val=args.b,tot_epoch=args.e,
                            train_dataset=tr_dataset,neural_network=nn_network,loss_function=loss_fn,CUDA=args.c)

    # Runs the training functionality
    if args.save is not None:
        main_class.train(model_file=args.save,load_param=args.load)

    # Runs the testing functionality
    if args.load is not None:
        test_dataset = MNISTWrapper(augment=False, train_bool=False,root_value=os.getcwd()+'/')
        main_class.test(model_file=args.load,tr_ds=tr_dataset,test_ds=test_dataset)

if __name__ == '__main__':
    main()