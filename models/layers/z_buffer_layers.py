import os

import torch
from torch import nn

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points

torch.manual_seed(42)

class RasterizePointsXYsBlending(nn.Module):
    """
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options

    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(
        self,
        C=64,
        learn_feature=True,
        radius=1.5,
        size=256,
        points_per_pixel=8,
        opts=None,
    ):
        super().__init__()
        if learn_feature:
            default_feature = nn.Parameter(torch.randn(1, C, 1))
            self.register_parameter("default_feature", default_feature)
        else:
            default_feature = torch.zeros(1, C, 1)
            self.register_buffer("default_feature", default_feature)

        self.radius = radius
        self.size = size
        self.points_per_pixel = points_per_pixel
        self.opts = opts

    def forward(self, pts3D, src):
        bs = src.size(0)
        if len(src.size()) > 3:
            bs, c, w, _ = src.size()
            image_size = w

            pts3D = pts3D.permute(0, 2, 1)
            src = src.unsqueeze(2).repeat(1, 1, w, 1, 1).view(bs, c, -1)
        else:
            bs = src.size(0)
            image_size = self.size

        # Make sure these have been arranged in the same way
        assert pts3D.size(2) == 3
        assert pts3D.size(1) == src.size(2)  

        pts3D[:,:,1] = - pts3D[:,:,1]
        pts3D[:,:,0] = - pts3D[:,:,0]

        # Add on the default feature to the end of the src
        # src = torch.cat((src, self.default_feature.repeat(bs, 1, 1)), 2)

        radius = float(self.radius) / float(image_size) * 2.0

        pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))
        points_idx, _, dist = rasterize_points(
            pts3D, image_size, radius, self.points_per_pixel
        )

        if os.environ["DEBUG"]:
            print("Max dist: ", dist.max(), pow(radius, self.opts.rad_pow))

        dist = dist / pow(radius, self.opts.rad_pow)

        if os.environ["DEBUG"]:
            print("Max dist: ", dist.max())

        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .pow(self.opts.tau)
            .permute(0, 3, 1, 2)
        )

        if self.opts.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.opts.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.opts.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.weighted_sum_norm(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )

        return transformed_src_alphas
