# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
from torch.nn import init

from models.losses.gan_loss import DiscriminatorLoss

class BaseModel(nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        self.model = model

        self.opt = opt

        if opt.discriminator_losses:
            self.use_discriminator = True

            self.netD = DiscriminatorLoss(opt)

            if opt.isTrain:
                self.optimizer_D = torch.optim.Adam(
                    list(self.netD.parameters()),
                    lr=opt.lr_d,
                    betas=(opt.beta1, opt.beta2),
                )
                self.optimizer_G = torch.optim.Adam(
                    list(self.model.parameters()),
                    lr=opt.lr_g,
                    betas=(opt.beta1, opt.beta2),
                )
        else:
            self.use_discriminator = False
            self.optimizer_G = torch.optim.Adam(
                list(self.model.parameters()),
                lr=opt.lr_g,
                betas=(0.99, opt.beta2),
            )

        if opt.isTrain:
            self.old_lr = opt.lr

        if opt.init:
            self.init_weights()

    def init_weights(self, gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if self.opt.init == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif self.opt.init == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif self.opt.init == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif self.opt.init == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif self.opt.init == "orthogonal":
                    init.orthogonal_(m.weight.data)
                elif self.opt.init == "":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented"
                        % self.opt.init
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(self.opt.init, gain)

    def __call__(
        self, dataloader, isval=False, num_steps=1, return_batch=False
    ):
        """
        Main function call
        - dataloader: The sampler that choose data samples.
        - isval: Whether to train the discriminator etc.
        - num steps: not fully implemented but is number of steps in the discriminator for
        each in the generator
        - return_batch: Whether to return the input values
        """
        weight = 1.0 / float(num_steps)
        if isval:
            batch = next(dataloader)
            t_losses, output_images = self.model(batch)

            if self.opt.normalize_image:
                for k in output_images.keys():
                    if "Img" in k:
                        output_images[k] = 0.5 * output_images[k] + 0.5

            if return_batch:
                return t_losses, output_images, batch
            return t_losses, output_images

        self.optimizer_G.zero_grad()
        if self.use_discriminator:
            all_output_images = []
            for j in range(0, num_steps):
                t_losses, output_images = self.model(next(dataloader))
                g_losses = self.netD.run_generator_one_step(
                    output_images["PredImg"], output_images["OutputImg"]
                )
                (
                    g_losses["Total Loss"] / weight
                    + t_losses["Total Loss"] / weight
                ).mean().backward()
                all_output_images += [output_images]
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            for step in range(0, num_steps):
                d_losses = self.netD.run_discriminator_one_step(
                    all_output_images[step]["PredImg"],
                    all_output_images[step]["OutputImg"],
                )
                (d_losses["Total Loss"] / weight).mean().backward()
            # Apply orthogonal regularization from BigGan
            self.optimizer_D.step()

            g_losses.pop("Total Loss")
            d_losses.pop("Total Loss")
            t_losses.update(g_losses)
            t_losses.update(d_losses)
        else:
            for step in range(0, num_steps):
                t_losses, output_images = self.model(next(dataloader))
                (t_losses["Total Loss"] / weight).mean().backward()
            self.optimizer_G.step()

        if self.opt.normalize_image:
            for k in output_images.keys():
                if "Img" in k:
                    output_images[k] = 0.5 * output_images[k] + 0.5

        return t_losses, output_images
