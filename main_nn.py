# Anant Bhargava, abharga7, HW3
import torchvision.transforms as transforms
import logging
import torchvision
from torchvision.transforms import ToTensor,Scale
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from caps_net import CapsuleNetwork
from util_class import main_run
def main():
    # Main Parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Part 1 A, CV HW 3, abharga7, Uses BCE Loss Function')
    parser.add_argument("--load", type=str, default=None,help='File to load with the weights')
    parser.add_argument("--aug", type=str, default=None,help='Whether augmented images should be used for training')
    parser.add_argument("--save", type=str, default=None,help='File to save the weights')
    parser.add_argument("--log_level", type=str, default='INFO',help='Current Logging Level')
    parser.add_argument("--num_epochs", type=int, default=10,help='Number of epochs')
    parser.add_argument("--batch_size", type=int, default=50,help='Number of image pairs in batch')
    parser.add_argument("--learning_rate", type=float, default=1e-6,help='Leaning Rate')
    args = parser.parse_args()
    # desired_transform = transforms.Compose([])

    # Setting up logger
    logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s',level=args.log_level)
    if args.aug is not None:
        logging.WARNING('AUGMENTATION NOT IMPLEMENTED')
        # # Setting up train dataset loading
        # train_dataset = torchvision.datasets.MNIST()
    else:
        # Setting up train dataset loading
        # train_dataset = two_image_loader(dir_name='./lfw', load_file='train.txt', trsfm_compose=desired_transform)
    train_dataset = torchvision.datasets.MNIST()
    # Main class to use
    main_class = main_run(l_r_val=args.learning_rate,batch_size_val=args.batch_size,tot_epoch=args.num_epochs,
                                train_loader=train_dataset,neural_network=CapsuleNetwork())

    # Runs the training functionality
    if args.save is not None:
        main_class.train(model_file=args.save,load_param=args.load)

    # Runs the testing functionality
    if args.load is not None:
        main_class.test(file_name=args.load)

if __name__ == '__main__':
    main()