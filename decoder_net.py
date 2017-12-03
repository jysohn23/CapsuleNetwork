"""
Decoder Network for the Capsule Net Architecture
"""

import torch
from torch.autograd import Variable


def mask(digit_caps_output, CUDA):
    caps_out_lens = torch.sqrt((digit_caps_output**2).sum(dim=2))
    _, max_vec_idx = caps_out_lens.max(dim=1)
    max_vec_idx = max_vec_idx.data
    batch_size = digit_caps_output.size(0)
    mask_vec = [None] * batch_size
    for i in range(batch_size):
        batch_sample = digit_caps_output[i]
        if CUDA:
            var = Variable(torch.zeros(batch_sample.size())).cuda()
        else:
            var = Variable(torch.zeros(batch_sample.size()))
        max_caps_idx = max_vec_idx[i]
        var[max_caps_idx] = batch_sample[max_caps_idx]
        mask_vec[i] = var
    return torch.stack(mask_vec, dim=0)


class DecoderNet(torch.nn.Module):
    def __init__(self, num_classes, output_unit_size, CUDA, fully_conn1_size=512, fully_conn2_size=1024, recon_size=784):
        super(DecoderNet, self).__init__()
        self.CUDA = CUDA
        self.fully_conn1 = torch.nn.Linear(num_classes * output_unit_size, fully_conn1_size)
        self.fully_conn2 = torch.nn.Linear(fully_conn1_size, fully_conn2_size)
        self.fully_conn3 = torch.nn.Linear(fully_conn2_size, recon_size)
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, digit_caps_output):
        masked_caps = mask(digit_caps_output, self.CUDA)
        img_recons = masked_caps.view(digit_caps_output.size(0), -1)
        relu1 = self.relu(self.fully_conn1(img_recons))
        relu2 = self.relu(self.fully_conn2(relu1))
        return self.sigmoid(self.fully_conn3(relu2))
