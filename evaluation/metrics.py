# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from models.losses.ssim import ssim

# The SSIM metric
def ssim_metric(img1, img2, mask=None):
    return ssim(img1, img2, mask=mask, size_average=False)


# The PSNR metric
def psnr(img1, img2, mask=None):
    b = img1.size(0)
    if not (mask is None):
        b = img1.size(0)
        mse_err = (img1 - img2).pow(2) * mask
        mse_err = mse_err.view(b, -1).sum(dim=1) / (
            3 * mask.view(b, -1).sum(dim=1).clamp(min=1)
        )
    else:
        mse_err = (img1 - img2).pow(2).view(b, -1).mean(dim=1)

    psnr = 10 * (1 / mse_err).log10()
    return psnr


# The perceptual similarity metric
def perceptual_sim(img1, img2, vgg16):
    # First extract features
    dist = vgg16(img1 * 2 - 1, img2 * 2 - 1)

    return dist
