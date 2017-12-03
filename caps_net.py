"""
Capsule Network Implementation
"""

import torch
from decoder_net import DecoderNet
from conv_layer import ConvLayer
from caps_layer import PrimaryCapsLayer, DigitCapsLayer


class CapsuleNetwork(torch.nn.Module):
    """
    Capsule Network Implementation extending the PyTorch Module class
    """
    def __init__(self, img_width, img_height, img_channel, num_conv_input_channels, num_conv_output_channels,
                 num_prim_units,  prim_unit_size, num_classes, output_unit_size, num_routing,
                 CUDA, conv_kernel_size, prim_kernel_size, prim_output_channels):
        """
        Constructor for CapsuleNetwork class

        :param img_width:                   Width of image (ex. MNIST=28)
        :param img_height:                  Height of image (ex. MNIST=28)
        :param img_channel:                 Number Channels per image (ex. MNIST=1)
        :param num_conv_input_channels:     Number of input channels for ConvLayer (ex. MNIST=1)
        :param num_conv_output_channels:    Number of output channels from ConvLayer (ex. MNIST=256)
        :param num_prim_units:              Number of primary unit (ex. MNIST=8)
        :param prim_unit_size:              Number of input channels of DigitCapsLayer (ex. MNIST=1152)
        :param num_classes:                 Number of classifications (ex. MNIST=10)
        :param output_unit_size:            Size of the output unit from DigitCapsLayer (ex. MNIST=16)
        :param num_routing:                 Number of iterations for the CapsNet routing mechanism
        :param CUDA:                        True if running on GPU, False otherwise
        :param conv_kernel_size:            Kernel length/width for ConvLayer (ex. MNIST=9)
        :param prim_kernel_size:            Kernel length/width for PrimaryCapsLayer (ex. MNIST=9)
        :param prim_output_channels:        Number of output channels of PrimaryCapsLayer (ex. MNIST=32)
        """
        super(CapsuleNetwork, self).__init__()
        self.CUDA = CUDA
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.num_classes = num_classes
        self.output_unit_size = output_unit_size
        self.nn = torch.nn.Sequential(
            ConvLayer(input_channels=num_conv_input_channels,
                      output_channels=num_conv_output_channels,
                      kernel_size=conv_kernel_size),
            PrimaryCapsLayer(input_channels=num_conv_output_channels,
                             num_unit=num_prim_units,
                             kernel_size=prim_kernel_size,
                             output_channels=prim_output_channels),
            DigitCapsLayer(input_unit=num_prim_units,
                           input_channels=prim_unit_size,
                           num_unit=num_classes,
                           unit_size=output_unit_size,
                           num_routes=num_routing,
                           CUDA=CUDA)
        )
        self.decoder = None

    def get_decoder(self):
        if self.decoder is None:
            self.decoder = DecoderNet(num_classes=self.num_classes, output_unit_size=self.output_unit_size, CUDA=self.CUDA)
        return self.decoder

    def forward(self, *input):
        return self.nn(input[0])

