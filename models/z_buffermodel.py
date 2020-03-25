# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from models.losses.synthesis import SynthesisLoss
from models.networks.architectures import Unet
from models.networks.utilities import get_decoder, get_encoder
from models.projection.z_buffer_manipulator import PtsManipulator


class ZbufferModelPts(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # ENCODER
        # Encode features to a given resolution
        self.encoder = get_encoder(opt)

        # POINT CLOUD TRANSFORMER
        # REGRESS 3D POINTS
        self.pts_regressor = Unet(channels_in=3, channels_out=1, opt=opt)

        if "modifier" in self.opt.depth_predictor_type:
            self.modifier = Unet(channels_in=64, channels_out=64, opt=opt)

        # 3D Points transformer
        if self.opt.use_rgb_features:
            self.pts_transformer = PtsManipulator(opt.W, C=3, opt=opt)
        else:
            self.pts_transformer = PtsManipulator(opt.W, opt=opt)

        self.projector = get_decoder(opt)

        # LOSS FUNCTION
        # Module to abstract away the loss function complexity
        self.loss_function = SynthesisLoss(opt=opt)

        self.min_tensor = self.register_buffer("min_z", torch.Tensor([0.1]))
        self.max_tensor = self.register_buffer(
            "max_z", torch.Tensor([self.opt.max_z])
        )
        self.discretized = self.register_buffer(
            "discretized_zs",
            torch.linspace(self.opt.min_z, self.opt.max_z, self.opt.voxel_size),
        )

    def forward(self, batch):
        """ Forward pass of a view synthesis model with a voxel latent field.
        """
        # Input values
        input_img = batch["images"][0]
        output_img = batch["images"][-1]

        if "depths" in batch.keys():
            depth_img = batch["depths"][0]

        # Camera parameters
        K = batch["cameras"][0]["K"]
        K_inv = batch["cameras"][0]["Kinv"]

        input_RT = batch["cameras"][0]["P"]
        input_RTinv = batch["cameras"][0]["Pinv"]
        output_RT = batch["cameras"][-1]["P"]
        output_RTinv = batch["cameras"][-1]["Pinv"]

        if torch.cuda.is_available():
            input_img = input_img.cuda()
            output_img = output_img.cuda()
            if "depths" in batch.keys():
                depth_img = depth_img.cuda()

            K = K.cuda()
            K_inv = K_inv.cuda()

            input_RT = input_RT.cuda()
            input_RTinv = input_RTinv.cuda()
            output_RT = output_RT.cuda()
            output_RTinv = output_RTinv.cuda()

        if self.opt.use_rgb_features:
            fs = input_img
        else:
            fs = self.encoder(input_img)

        # Regressed points
        if not (self.opt.use_gt_depth):
            if not('use_inverse_depth' in self.opt) or not(self.opt.use_inverse_depth):
                regressed_pts = (
                    nn.Sigmoid()(self.pts_regressor(input_img))
                    * (self.opt.max_z - self.opt.min_z)
                    + self.opt.min_z
                )

            else:
                # Use the inverse for datasets with landscapes, where there
                # is a long tail on the depth distribution
                depth = self.pts_regressor(input_img)
                regressed_pts = 1. / (nn.Sigmoid()(depth) * 10 + 0.01)
        else:
            regressed_pts = depth_img

        gen_fs = self.pts_transformer.forward_justpts(
            fs,
            regressed_pts,
            K,
            K_inv,
            input_RT,
            input_RTinv,
            output_RT,
            output_RTinv,
        )

        if "modifier" in self.opt.depth_predictor_type:
            gen_fs = self.modifier(gen_fs)

        gen_img = self.projector(gen_fs)

        # And the loss
        loss = self.loss_function(gen_img, output_img)

        if self.opt.train_depth:
            depth_loss = nn.L1Loss()(regressed_pts, depth_img)
            loss["Total Loss"] += depth_loss
            loss["depth_loss"] = depth_loss

        return (
            loss,
            {
                "InputImg": input_img,
                "OutputImg": output_img,
                "PredImg": gen_img,
                "PredDepth": regressed_pts,
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

            RTs = [RT.cuda() for RT in RTs]
            identity = (
                torch.eye(4).unsqueeze(0).repeat(input_img.size(0), 1, 1).cuda()
            )

        fs = self.encoder(input_img)
        regressed_pts = (
            nn.Sigmoid()(self.pts_regressor(input_img))
            * (self.opt.max_z - self.opt.min_z)
            + self.opt.min_z
        )

        # Now rotate
        gen_imgs = []
        for RT in RTs:
            torch.manual_seed(
                0
            )  # Reset seed each time so that noise vectors are the same
            gen_fs = self.pts_transformer.forward_justpts(
                fs, regressed_pts, K, K_inv, identity, identity, RT, None
            )

            # now create a new image
            gen_img = self.projector(gen_fs)

            gen_imgs += [gen_img]

        if return_depth:
            return gen_imgs, regressed_pts

        return gen_imgs
