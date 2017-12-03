"""
Conv1 Layer Implementation
"""

import torch

class Conv1Layer(torch.nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super(Conv1Layer, self).__init__()
        self.nn = torch.nn.Sequential(torch.nn.Conv2d(in_channels=input_channels,
                                                      out_channels=output_channels,
                                                      kernel_size=kernel_size,
                                                      stride=stride),
                                      torch.nn.ReLU(inplace=True))

    def forward(self, *input):
        return self.nn(input[0])

