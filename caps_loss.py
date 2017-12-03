"""
Loss Function(s) for the Capsule Network Implementation
"""


import torch


class CapsuleLoss(torch.nn.Module):

    def __init__(self, margin_loss, reconstruction_loss, regularization_scale, CUDA, decoder):
        super(CapsuleLoss, self).__init__()
        self.margin_loss = margin_loss
        self.reconstruction_loss = reconstruction_loss
        self.regularization_scale = regularization_scale
        self.CUDA = CUDA
        self.decoder = decoder

    def __mean_margin_loss(self, digit_caps_output, labels):
        batch_size = digit_caps_output.size(0)
        norm = torch.sqrt((digit_caps_output**2).sum(dim=2, keepdim=True))

        # if self.CUDA:
        #     clamp_zero = Variable(torch.zeros(1)).cuda()
        # else:
        #     clamp_zero = Variable(torch.zeros(1))

        # TODO: BUG DANGER ZONE
        max_true = torch.clamp(0.9 - norm, min=0).view(batch_size, -1)**2
        max_false = torch.clamp(norm - 0.1, min=0).view(batch_size, -1)**2
        margin_losses = labels * max_true + 0.5 * (1.0 - labels) * max_false
        return margin_losses.sum(dim=1).mean()

    def __mean_recon_loss(self, digit_caps_output, labels, imgs):
        recon = self.decoder(digit_caps_output, labels)
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, -1)
        sqr_err = (recon - imgs)**2
        return sqr_err.sum(dim=1).mean()

    def forward(self, imgs, digit_caps_output, labels):
        margin_loss = self.__mean_margin_loss(digit_caps_output, labels)
        reconstruction_loss = self.__mean_recon_loss(digit_caps_output, labels, imgs)
        total_loss = margin_loss + reconstruction_loss * self.regularization_scale
        return total_loss, margin_loss, reconstruction_loss

