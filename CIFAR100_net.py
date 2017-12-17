from torch import nn
from conv_layer import ConvLayer
from caps_layer import PrimaryCapsLayer, DigitCapsLayer

class CIFAR100nn(nn.Module):
    """
    CIFAR 100 module
    """
    def __init__(self, CUDA):
        """
        Constructor for CapsuleNetwork class
        """
        super(CIFAR100nn, self).__init__()
        self.img_width = 32
        self.img_height = 32
        self.img_channel = 3
        self.CUDA = CUDA
        self.nn = nn.Sequential(
            ConvLayer(input_channels=3, output_channels=256, kernel_size=9),
            PrimaryCapsLayer(input_channels=256, num_unit=8, kernel_size=9, output_channels=32),
            DigitCapsLayer(input_unit=8, input_channels=2048, num_unit=100, unit_size=8, num_routes=3, CUDA=CUDA)
        )
        self.decoder = None

    def forward(self, *input):
        return self.nn(input[0])

    def get_decoder(self):
        return self.decoder