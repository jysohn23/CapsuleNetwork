# Main Function
import logging
import torchvision
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from caps_net import CapsuleNetwork
from util_class import main_run
import torchvision.transforms as transforms

def main():
    # Main Parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='')
    parser.add_argument("--load", type=str, default=None,help='File to load with the weights')
    parser.add_argument("--aug", type=str, default=None,help='Whether augmented images should be used for training')
    parser.add_argument("--save", type=str, default=None,help='File to save the weights')
    parser.add_argument("--log_level", type=str, default='INFO',help='Current Logging Level')
    parser.add_argument("--num_epochs", type=int, default=10,help='Number of epochs')
    parser.add_argument("--batch_size", type=int, default=50,help='Number of image pairs in batch')
    parser.add_argument("--learning_rate", type=float, default=1e-6,help='Leaning Rate')
    args = parser.parse_args()

    # Setting up logger
    logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s',level=args.log_level)
    # Getting the desired transform
    if args.aug is None:
        logging.info('No augmentation')
        desired_transform = transforms.Compose([transforms.ToTensor()])
    else:
        logging.info('Performing Augmentation')
        desired_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomSizedCrop(size=32),
                                                transforms.ToTensor()])
    tr_dataset = torchvision.datasets.MNIST(transform=desired_transform,root='~')
    test_dataset = torchvision.datasets.MNIST(train=False,transform=desired_transform,root='~')
    # Main class to use
    main_class = main_run(l_r_val=args.learning_rate,batch_size_val=args.batch_size,tot_epoch=args.num_epochs,
                                train_loader=tr_dataset,neural_network=CapsuleNetwork())

    # Runs the training functionality
    if args.save is not None:
        main_class.train(model_file=args.save,load_param=args.load)

    # Runs the testing functionality
    if args.load is not None:
        main_class.test(model_file=args.load,tr_ds=tr_dataset,test_ds=test_dataset)

if __name__ == '__main__':
    main()