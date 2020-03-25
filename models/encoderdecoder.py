# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.synthesis import SynthesisLoss


class CollapseLayer(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnCollapseLayer(nn.Module):
    def __init__(self, C, W, H):
        super().__init__()
        self.C = C
        self.W = W
        self.H = H

    def forward(self, input):
        return input.view(input.size(0), self.C, self.W, self.H)


class ViewAppearanceFlow(nn.Module):
    """
    View Appearance Flow based on the corresponding paper.
    """

    def __init__(self, opt):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),  # 128
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 64
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # 32
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),  # 16
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),  # 8
            nn.Conv2d(256, 512, 3, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),  # 4
            CollapseLayer(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.BatchNorm2d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.BatchNorm2d(4096),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4096 + 256, 4096),
            nn.ReLU(),
            nn.BatchNorm2d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.BatchNorm2d(4096),
            UnCollapseLayer(64, 8, 8),
            nn.Conv2d(64, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 2, 3, 1, padding=1),
            nn.Tanh(),
        )

        self.angle_transformer = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.loss_function = SynthesisLoss(opt=opt)

        self.opt = opt

    def forward(self, batch):
        input_img = batch["images"][0]
        output_img = batch["images"][-1]

        input_RTinv = batch["cameras"][0]["Pinv"]
        output_RT = batch["cameras"][-1]["P"]

        if torch.cuda.is_available():
            input_img = input_img.cuda()
            output_img = output_img.cuda()

            input_RTinv = input_RTinv.cuda()
            output_RT = output_RT.cuda()

        RT = input_RTinv.bmm(output_RT)[:, 0:3, :]

        # Now transform the change in angle
        fs = self.encoder(input_img)
        fs_angle = self.angle_transformer(RT.view(RT.size(0), -1))

        # And concatenate
        fs = torch.cat((fs, fs_angle), 1)
        sampler = self.decoder(fs)
        gen_img = F.grid_sample(input_img, sampler.permute(0, 2, 3, 1))

        # And the loss
        loss = self.loss_function(gen_img, output_img)

        # And return
        return (
            loss,
            {
                "InputImg": input_img,
                "OutputImg": output_img,
                "PredImg": gen_img,
            },
        )

    def forward_angle(self, batch, RTs, return_depth=False):
        # Input values
        input_img = batch["images"][0]

        # Camera parameters
        K = batch["cameras"][0]["K"]
        K_inv = batch["cameras"][0]["Kinv"]

        if torch.cuda.is_available():
            input_img = input_img.cuda()

            K = K.cuda()
            K_inv = K_inv.cuda()

            RTs = [RT[:, 0:3, :].cuda() for RT in RTs]

        fs = self.encoder(input_img)
        # Now rotate
        gen_imgs = []
        for i, RT in enumerate(RTs):
            torch.manual_seed(
                0
            )  # Reset seed each time so that noise vectors are the same
            fs_angle = self.angle_transformer(RT.view(RT.size(0), -1))

            # And concatenate
            fs_new = torch.cat((fs, fs_angle), 1)
            sampler = self.decoder(fs_new)
            gen_img = F.grid_sample(input_img, sampler.permute(0, 2, 3, 1))

            gen_imgs += [gen_img]

        if return_depth:
            return gen_imgs, torch.zeros(fs.size(0), 1, 256, 256)

        return gen_imgs


class Tatarchenko(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),  # 128
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),  # 64
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),  # 32
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),  # 16
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),  # 8
            nn.Conv2d(256, 512, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),  # 4
            CollapseLayer(),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4096),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4096),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4096 + 64, 4096),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4096),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4096),
            UnCollapseLayer(64, 8, 8),
            nn.Conv2d(64, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 3, 3, 1, padding=1),
            nn.Tanh(),
        )

        self.angle_transformer = nn.Sequential(
            nn.Linear(12, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
        )

        self.loss_function = SynthesisLoss(opt=opt)

        self.opt = opt

    def forward(self, batch):
        input_img = batch["images"][0]
        output_img = batch["images"][-1]

        input_RTinv = batch["cameras"][0]["Pinv"]
        output_RT = batch["cameras"][-1]["P"]

        if torch.cuda.is_available():
            input_img = input_img.cuda()
            output_img = output_img.cuda()

            input_RTinv = input_RTinv.cuda()
            output_RT = output_RT.cuda()

        RT = input_RTinv.bmm(output_RT)[:, 0:3, :]

        # Now transform the change in angle
        fs = self.encoder(input_img)
        fs_angle = self.angle_transformer(RT.view(RT.size(0), -1))

        # And concatenate
        fs = torch.cat((fs, fs_angle), 1)
        gen_img = self.decoder(fs)

        loss = self.loss_function(gen_img, output_img)

        # And return
        return (
            loss,
            {
                "InputImg": input_img,
                "OutputImg": output_img,
                "PredImg": gen_img,
            },
        )
