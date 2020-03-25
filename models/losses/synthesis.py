# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from models.losses.ssim import ssim
from models.networks.architectures import VGG19


class SynthesisLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # Get the losses
        print(opt.losses)
        print(zip(*[l.split("_") for l in opt.losses]))
        lambdas, loss_names = zip(*[l.split("_") for l in opt.losses])
        lambdas = [float(l) for l in lambdas]

        loss_names += ("PSNR", "SSIM")

        self.lambdas = lambdas
        self.losses = nn.ModuleList(
            [self.get_loss_from_name(loss_name) for loss_name in loss_names]
        )

    def get_loss_from_name(self, name):
        if name == "l1":
            loss = L1LossWrapper()
        elif name == "content":
            loss = PerceptualLoss(self.opt)
        elif name == "PSNR":
            loss = PSNR()
        elif name == "SSIM":
            loss = SSIM()

        if torch.cuda.is_available():
            return loss.cuda()

    def forward(self, pred_img, gt_img):
        losses = [loss(pred_img, gt_img) for loss in self.losses]

        loss_dir = {}
        for i, l in enumerate(losses):
            if "Total Loss" in l.keys():
                if "Total Loss" in loss_dir.keys():
                    loss_dir["Total Loss"] = (
                        loss_dir["Total Loss"]
                        + l["Total Loss"] * self.lambdas[i]
                    )
                else:
                    loss_dir["Total Loss"] = l["Total Loss"]

            loss_dir = dict(l, **loss_dir)  # Have loss_dir override l

        return loss_dir


class PSNR(nn.Module):
    def forward(self, pred_img, gt_img):
        bs = pred_img.size(0)
        mse_err = (pred_img - gt_img).pow(2).sum(dim=1).view(bs, -1).mean(dim=1)

        psnr = 10 * (1 / mse_err).log10()
        return {"psnr": psnr.mean()}


class SSIM(nn.Module):
    def forward(self, pred_img, gt_img):
        return {"ssim": ssim(pred_img, gt_img)}


# Wrapper of the L1Loss so that the format matches what is expected
class L1LossWrapper(nn.Module):
    def forward(self, pred_img, gt_img):
        err = nn.L1Loss()(pred_img, gt_img)
        return {"L1": err, "Total Loss": err}


# Implementation of the perceptual loss to enforce that a
# generated image matches the given image.
# Adapted from SPADE's implementation
# (https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py)
class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.model = VGG19(
            requires_grad=False
        )  # Set to false so that this part of the network is frozen
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, pred_img, gt_img):
        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)

        # Collect the losses at multiple layers (need unsqueeze in
        # order to concatenate these together)
        loss = 0
        for i in range(0, len(gt_fs)):
            loss += self.weights[i] * self.criterion(pred_fs[i], gt_fs[i])

        return {"Perceptual": loss, "Total Loss": loss}
