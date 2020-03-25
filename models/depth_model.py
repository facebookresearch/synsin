# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.projection import depth_manipulator

EPS = 1e-4


class Model(nn.Module):
    """
    Model for transforming by depth. Used only in evaluation, not training.
    As a result, use a brute force method for splatting points. This is 
    improved in the z_buffermodel.py code to allow for differentiable rendering.
    """
    def __init__(self, opt):
        super(Model, self).__init__()
        self.use_z = opt.use_z
        self.use_alpha = opt.use_alpha
        self.opt = opt

        # REFINER
        # Refine the projected depth

        num_inputs = 3

        opt.num_inputs = num_inputs

        # PROJECTION
        # Project according to the predicted depth
        self.projector = depth_manipulator.DepthManipulator(W=opt.W)


    def forward(self, batch):
        """ Forward pass of a view synthesis model with a predicted depth.
        """
        # Input values
        input_img = batch["images"][0]
        depth_img = batch["depths"][0]
        output_img = batch["images"][-1]

        # Camera parameters
        K = batch["cameras"][0]["K"]
        K_inv = batch["cameras"][0]["Kinv"]

        input_RTinv = batch["cameras"][0]["Pinv"]
        output_RT = batch["cameras"][-1]["P"]

        if torch.cuda.is_available():
            input_img = input_img.cuda()
            depth_img = depth_img.cuda()
            output_img = output_img.cuda()

            K = K.cuda()
            K_inv = K_inv.cuda()

            input_RTinv = input_RTinv.cuda()
            output_RT = output_RT.cuda()

        # Transform the image according to intrinsic parameters
        # and rotation and depth
        sampled_image = self.transform_perfimage(
            input_img, output_img, depth_img, K, K_inv, input_RTinv, output_RT
        )

        mask = (batch["depths"][1] < 10).float() * (
            batch["depths"][1] > EPS
        ).float()

        return (
            0,
            {
                "InputImg": input_img,
                "OutputImg": output_img,
                "Mask": mask.float(),
                "SampledImg": sampled_image[:, 0:3, :, :],
                "Diff Sampled": (
                    sampled_image[:, 0:3, :, :] - output_img
                ).abs(),
                "Depth": depth_img,
            },
        )

    def transform_perfimage(
        self, input_img, output_img, depth_img, K, K_inv, RTinv_cam1, RT_cam2
    ):
        """ Create a new view of an input image.
        Transform according to the output rotation/translation.
        """
        # Transform according to the depth projection
        sampler, _ = self.projector.project_zbuffer(
            depth=depth_img,
            K=K,
            K_inv=K_inv,
            RTinv_cam1=RTinv_cam1,
            RT_cam2=RT_cam2,
        )

        # Sample image according to this sampler

        mask = ((sampler > -1).float() * (sampler < 1).float()).min(
            dim=1, keepdim=True
        )[0]
        mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = (mask > 0).float()

        sampled_image = output_img * mask

        return sampled_image
