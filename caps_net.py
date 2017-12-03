"""
Capsule Network Implementation
"""

import torch
from conv1_layer import Conv1Layer
from caps_layer import PrimaryCapsLayer, DigitCapsLayer


class CapsuleNetwork(torch.nn.Module):

    def __init__(self, img_width, img_height, img_channel, num_conv_input_channels, num_conv_output_channels,
                 num_prim_units,  prim_unit_size, num_classes, output_unit_size, num_routing,
                 CUDA, conv_kernel_size, prim_kernel_size, prim_output_channels):
        super(CapsuleNetwork, self).__init__()
        self.CUDA = CUDA
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.nn = torch.nn.Sequential(
            Conv1Layer(input_channels=num_conv_input_channels,
                       output_channels=num_conv_output_channels,
                       kernel_size=conv_kernel_size),
            PrimaryCapsLayer(input_channels=num_conv_output_channels,
                             num_unit=num_prim_units,
                             unit_size=prim_unit_size,
                             kernel_size=prim_kernel_size,
                             output_channels=prim_output_channels),
            DigitCapsLayer(input_unit=num_prim_units,
                           input_channels=prim_unit_size,
                           num_unit=num_classes,
                           unit_size=output_unit_size,
                           num_routes=num_routing,
                           CUDA=CUDA)
        )


    def forward(self, *input):
        return self.nn(input[0])

