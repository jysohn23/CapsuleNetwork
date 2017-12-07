"""
CIFAR 10 Implementation
"""

import torch.nn as nn
from conv_layer import ConvLayer
from caps_layer import PrimaryCapsLayer, DigitCapsLayer
from decoder_net import mask

class Decoder_Color(nn.Module):
    def __init__(self, CUDA):
        super(Decoder_Color, self).__init__()
        self.CUDA = CUDA
        self.module = nn.Sequential(nn.ConvTranspose2d(in_channels=10,out_channels=7,kernel_size=3,stride=3), #(80,16)
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(in_channels=7, out_channels=5, kernel_size=4,stride=2),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(in_channels=5, out_channels=4, kernel_size=4, stride=1),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=4, stride=1),
                                    nn.Tanh()
                                    )
    def forward(self, digit_caps_output, labels):
        # Get a mask and then get the run over the network
        masked = mask(digit_caps_output, self.CUDA).view(digit_caps_output.size(0),10,4,4)
        return self.module(masked).view(digit_caps_output.size(0), -1)



class CIFAR10nn(nn.Module):
    """
    CIFAR 10 module
    """
    def __init__(self, CUDA):
        """
        Constructor for CapsuleNetwork class
        """
        super(CIFAR10nn, self).__init__()
        self.img_width = 32
        self.img_height = 32
        self.img_channel = 3
        self.CUDA = CUDA
        self.nn = nn.Sequential(
            ConvLayer(input_channels=3, output_channels=256, kernel_size=9),
            PrimaryCapsLayer(input_channels=256, num_unit=8, kernel_size=9, output_channels=32),
            DigitCapsLayer(input_unit=8, input_channels=2048, num_unit=10, unit_size=16, num_routes=3, CUDA=CUDA)
        )
        self.decoder = Decoder_Color(CUDA=self.CUDA)

    def forward(self, *input):
        return self.nn(input[0])

    def get_decoder(self):
        return self.decoder

