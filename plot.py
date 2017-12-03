# Plotting utility
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt

def plot_graph(file_name,plot_title,out_file):
    main_arr=np.loadtxt(file_name)
    plt.plot(main_arr)
    plt.ylabel('Loss Function')
    plt.xlabel('Iteration')
    plt.title(plot_title)
    plt.savefig(out_file)

def main():
    # Main Parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Plotting utility')
    parser.add_argument("--input", type=str, default=None,help='File to load with the weights')
    parser.add_argument("--output", type=str, default=None,help='Whether augmented images should be used for training')
    parser.add_argument("--title", type=str, default=None,help='File to save the weights')
    args = parser.parse_args()

    if (args.input is not None) & (args.output is not None) & (args.title is not None):
        plot_graph(args.input,args.title,args.output)

if __name__ == '__main__':
    main()