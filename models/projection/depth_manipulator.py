# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-2


class DepthManipulator(nn.Module):
    """
    Use depth in order to naively manipulate an image. Simply splatter the
    depth into the new image such that the nearest point colours a pixel.

    Is not used for training; just for evaluation to determine visible/invisible
    regions.
    """
    def __init__(self, W=256):
        super(DepthManipulator, self).__init__()
        # Set up default grid using clip coordinates (e.g. between [-1, 1])
        xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, W))
        xs = xs.reshape(1, W, W)
        ys = ys.reshape(1, W, W)
        xys = np.vstack((xs, ys, -np.ones(xs.shape), np.ones(xs.shape)))

        self.grid = torch.Tensor(xys).unsqueeze(0)

        if torch.cuda.is_available():
            self.grid = self.grid.cuda()

    def homogenize(self, xys):
        assert xys.size(1) <= 3
        ones = torch.ones(xys.size(0), 1, xys.size(2)).to(xys.device)

        return torch.cat((xys, ones), 1)

    def project_zbuffer(self, depth, K, K_inv, RTinv_cam1, RT_cam2):
        """ Determine the sampler that comes from projecting
        the given depth according to the given camera parameters.
        """
        bs, _, w, h = depth.size()

        # Obtain unprojected coordinates
        orig_xys = self.grid.to(depth.device).repeat(bs, 1, 1, 1).detach()
        xys = orig_xys * depth
        xys[:, -1, :] = 1

        xys = xys.view(bs, 4, -1)

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(xys)

        # Transform into world coordinates
        RT = RT_cam2.bmm(RTinv_cam1)
        wrld_X = RT.bmm(cam1_X)

        # And intrinsics
        xy_proj = K.bmm(wrld_X)

        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()
        sampler = xy_proj[:, 0:2, :] / -xy_proj[:, 2:3, :]
        sampler[mask.repeat(1, 2, 1)] = -10
        sampler[:, 1, :] = -sampler[:, 1, :]
        sampler[:, 0, :] = sampler[:, 0, :]

        with torch.no_grad():
            print(
                "Warning : not backpropagating through the "
                + "projection -- is this what you want??"
            )
            tsampler = (sampler + 1) * 128
            tsampler = tsampler.view(bs, 2, -1)
            zs, sampler_inds = xy_proj[:, 2:3, :].sort(
                dim=2, descending=True
            )  # Hack for how it's going to be understood by scatter: enforces that
            # nearer points are the ones rendered.
            bsinds = (
                torch.linspace(0, bs - 1, bs)
                .long()
                .unsqueeze(1)
                .repeat(1, w * h)
                .to(sampler.device)
                .unsqueeze(1)
            )

            xs = tsampler[bsinds, 0, sampler_inds].long()
            ys = tsampler[bsinds, 1, sampler_inds].long()
            mask = (tsampler < 0) | (tsampler > 255)
            mask = mask.float().max(dim=1, keepdim=True)[0] * 4

            xs = xs.clamp(min=0, max=255)
            ys = ys.clamp(min=0, max=255)

            bilinear_sampler = torch.zeros(bs, 2, w, h).to(sampler.device) - 2
            orig_xys = orig_xys[:, :2, :, :].view((bs, 2, -1))
            bilinear_sampler[bsinds, 0, ys, xs] = (
                orig_xys[bsinds, 0, sampler_inds] + mask
            )
            bilinear_sampler[bsinds, 1, ys, xs] = (
                -orig_xys[bsinds, 1, sampler_inds] + mask
            )

        return bilinear_sampler, -xy_proj[:, 2:3, :].view(bs, 1, w, h)
