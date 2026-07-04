import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def range_compressor(hdr_img, mu=5000):
    return torch.log1p(mu * hdr_img) / math.log1p(mu)


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super().__init__()
        self.mu = mu
        self.loss = nn.L1Loss()

    def forward(self, pred, label):
        return self.loss(range_compressor(pred, self.mu), range_compressor(label, self.mu))


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),
            vgg[4:9].eval(),
            vgg[9:16].eval(),
            vgg[16:23].eval(),
        ])
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=(0, 1, 2, 3), style_layers=()):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = F.interpolate(input, mode="bilinear", size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode="bilinear", size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                loss += F.l1_loss(act_x @ act_x.permute(0, 2, 1), act_y @ act_y.permute(0, 2, 1))
        return loss


class JointReconPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1MuLoss(mu)
        self.loss_vgg = VGGPerceptualLoss(resize=False)

    def forward(self, input, target):
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        return self.loss_recon(input, target) + self.alpha * self.loss_vgg(input_mu, target_mu)
