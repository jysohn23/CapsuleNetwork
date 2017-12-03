"""
Capsule Layer Implementation
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Utils
def squash(unit, dim):
    norm_sqr = (unit**2).sum(dim=dim, keepdim=True)
    return (norm_sqr / (1 + norm_sqr)) * (unit / torch.sqrt(norm_sqr))


def soft_max_nd(routing_logits, dim):
    logits_size = routing_logits.size()
    logits_t = routing_logits.transpose(dim, len(logits_size) - 1)
    logits_t_size = logits_t.size()
    logits_2d = logits_t.contiguous().view(-1, logits_t_size[-1])
    softmax_2d = F.softmax(logits_2d)
    softmax_nd = softmax_2d.view(*logits_t_size)
    return softmax_nd.transpose(dim, len(logits_size) - 1)


class CapsLayer(torch.nn.Module):

    def __init__(self, input_unit, input_channels, num_unit, unit_size):
        super(CapsLayer, self).__init__()
        self.input_unit = input_unit
        self.input_channels = input_channels
        self.num_unit = num_unit
        self.unit_size = unit_size

    def forward(self, *input):
        raise NotImplementedError


class PrimaryCapsLayer(CapsLayer):

    def __init__(self, input_unit=8, input_channels=1152, num_unit=10, unit_size=16,
                 kernel_size=9, stride=2, output_channels=32):
        super(PrimaryCapsLayer, self).__init__(input_unit, input_channels, num_unit, unit_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_channels = output_channels
        self.conv_list = torch.nn.ModuleList(
            [torch.nn.Conv2d(
                self.input_channels,
                self.output_channels,
                self.kernel_size,
                self.stride
            ) for _ in range(unit_size)]
        )

    def forward(self, *input):
        conv_out = input[0]
        unit = torch.stack([self.conv_list[i](conv_out) for i, _ in enumerate(self.conv_list)], dim=1)
        batch_size = conv_out.size(0)
        # Flatten
        unit = unit.view(batch_size, self.num_unit, -1)
        return squash(unit, 2)


class DigitCapsLayer(CapsLayer):

    def __init__(self, input_unit=8, input_channels=1152, num_unit=10, unit_size=16, num_routes=4, CUDA=True):
        super(DigitCapsLayer, self).__init__(input_unit, input_channels, num_unit, unit_size)
        self.num_routes = num_routes
        self.CUDA = CUDA
        # Random Initialization of Weights
        self.weights = torch.nn.Parameter(
            torch.randn(
                1,
                self.input_channels,
                self.num_unit,
                self.unit_size,
                self.input_unit
            )
        )

    def forward(self, *input):
        prim_caps_out = input[0]
        batch_size = prim_caps_out.size(0)
        prim_caps_out = prim_caps_out.transpose(1, 2)
        prim_caps_out = torch.stack([prim_caps_out] * self.num_unit, dim=2).unsqueeze(4)
        batch_weight = torch.cat([self.weights] * batch_size, dim=0)
        predict_vectors = torch.matmul(batch_weight, prim_caps_out)

        if self.CUDA:
            routing_logits = Variable(torch.zero(1, self.input_channels, self.num_unit, 1)).cuda()
        else:
            routing_logits = Variable(torch.zero(1, self.input_channels, self.num_unit, 1))

        for _ in range(self.num_routes):
            # Calculate coupling coefficients
            coup_coeff = soft_max_nd(routing_logits, dim=2)
            coup_coeff = torch.cat([coup_coeff] * batch_size, dim=0).unsqueeze(4)
            # Calculate total input to a capsule (weighted sum over all prediction vectors)
            sum_caps_out = (coup_coeff * predict_vectors).sum(dim=1, keepdim=True)
            # Squash the vector output of the capsule
            vec_out = squash(sum_caps_out, dim=3)
            vec_out_cat = torch.cat([vec_out] * self.input_channels, dim=1)
            # Calculate the agreement and update the initial routing logits
            agreement = torch.matmul(predict_vectors.transpose(3, 4), vec_out_cat).squeeze(4).mean(dim=0, keepdim=True)
            routing_logits += agreement

        return vec_out.squeeze(1)
