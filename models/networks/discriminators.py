"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Based on https://github.com/NVlabs/SPADE/blob/master/models/pix2pix_model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from models.layers.normalization import get_D_norm_layer


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            "Network [%s] was created."
            + "Total number of parameters: %.1f million. "
            "To see the architecture, do print(network)."
            % (type(self).__name__, num_params / 1000000)
        )

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "none":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented"
                        % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, gain)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.n_layers_D = 4
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_D_norm_layer(opt, opt.norm_D)
        sequence = [
            [
                nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, False),
            ]
        ]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [
                [
                    norm_layer(
                        nn.Conv2d(
                            nf_prev,
                            nf,
                            kernel_size=kw,
                            stride=stride,
                            padding=padw,
                        )
                    ),
                    nn.LeakyReLU(0.2, False),
                ]
            ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module("model" + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        # if opt.concat_discriminators:
        #     input_nc = opt.output_nc * 2
        # else:
        input_nc = opt.output_nc
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.netD_subarch = "n_layer"
        opt.num_D = 2
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module("discriminator_%d" % i, subnetD)
        if opt.isTrain:
            self.old_lr = opt.lr

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == "n_layer":
            netD = NLayerDiscriminator(opt)

            if torch.cuda.is_available():
                netD = netD.cuda()
        else:
            raise ValueError(
                "unrecognized discriminator subarchitecture %s" % subarch
            )
        return netD

    def downsample(self, input):
        return F.avg_pool2d(
            input,
            kernel_size=3,
            stride=2,
            padding=[1, 1],
            count_include_pad=False,
        )

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            print("update learning rate: %f -> %f" % (self.old_lr, new_lr))
            self.old_lr = new_lr
            new_lr_G = new_lr / 2
            new_lr_D = new_lr * 2
            return False, {"lr_D": new_lr_D, "lr_G": new_lr_G}

        else:
            return False, {"lr_D": new_lr, "lr_G": new_lr}

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss

        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


def define_D(opt):
    net = MultiscaleDiscriminator(opt)
    net.init_weights("xavier", 0.02)
    if torch.cuda.is_available():
        net = net.cuda()
    return net
