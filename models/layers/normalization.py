# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn.parameter import Parameter

def get_linear_layer(opt, bias=False):
    if "spectral" in opt.norm_G:
        linear_layer_base = lambda in_c, out_c: nn.utils.spectral_norm(
            nn.Linear(in_c, out_c, bias=bias)
        )
    else:
        linear_layer_base = lambda in_c, out_c: nn.Linear(
            in_c, out_c, bias=bias
        )

    return linear_layer_base


class LinearNoiseLayer(nn.Module):
    def __init__(self, opt, noise_sz=20, output_sz=32):
        """
        Class for adding in noise to the batch normalisation layer.
        Based on the idea from BigGAN.
        """
        super().__init__()
        self.noise_sz = noise_sz

        linear_layer = get_linear_layer(opt, bias=False)

        self.gain = linear_layer(noise_sz, output_sz)
        self.bias = linear_layer(noise_sz, output_sz)

        self.bn = bn(output_sz)

        self.noise_sz = noise_sz

    def forward(self, x):
        noise = torch.randn(x.size(0), self.noise_sz).to(x.device)

        # Predict biases/gains for this layer from the noise
        gain = (1 + self.gain(noise)).view(noise.size(0), -1, 1, 1)
        bias = self.bias(noise).view(noise.size(0), -1, 1, 1)

        xp = self.bn(x, gain=gain, bias=bias)
        return xp


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_D_norm_layer(opt, norm_type="instance"):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, "out_channels"):
            return getattr(layer, "out_channels")
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith("spectral"):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len("spectral") :]

        if subnorm_type == "none" or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, "bias", None) is not None:
            delattr(layer, "bias")
            layer.register_parameter("bias", None)

        if subnorm_type == "batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)

        elif subnorm_type == "instance":
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError(
                "normalization layer %s is not recognized" % subnorm_type
            )

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# BatchNorm layers are taken from the BigGAN code base.
# https://github.com/ajbrock/BigGAN-PyTorch/blob/a5557079924c3070b39e67f2eaea3a52c0fb72ab/layers.py
# Distributed under the MIT licence.

# Normal, non-class-conditional BN
class BatchNorm_StandingStats(nn.Module):
    def __init__(self, output_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.register_parameter("gain", Parameter(torch.ones(output_size)))
        self.register_parameter("bias", Parameter(torch.zeros(output_size)))
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.bn = bn(output_size, self.eps, self.momentum)

    def forward(self, x, y=None):
        gain = self.gain.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return self.bn(x, gain=gain, bias=bias)

class bn(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super().__init__()

        # momentum for updating stats
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("stored_mean", torch.zeros(num_channels))
        self.register_buffer("stored_var", torch.ones(num_channels))
        self.register_buffer("accumulation_counter", torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(
                x, gain, bias, return_mean_var=True, eps=self.eps
            )
            # If accumulating standing stats, increment them
            with torch.no_grad():
                if self.accumulate_standing:
                    self.stored_mean[:] = self.stored_mean + mean.data
                    self.stored_var[:] = self.stored_var + var.data
                    self.accumulation_counter += 1.0
                # If not accumulating standing stats, take running averages
                else:
                    self.stored_mean[:] = (
                        self.stored_mean * (1 - self.momentum)
                        + mean * self.momentum
                    )
                    self.stored_var[:] = (
                        self.stored_var * (1 - self.momentum) + var * self.momentum
                    )
            return out
        # If not in training mode, use the stored statistics
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            # If using standing stats, divide them by the accumulation counter
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = m2 - m ** 2
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)
