# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torchvision

from models.layers.blocks import ResNet_Block
from models.layers.normalization import BatchNorm_StandingStats
from models.networks.configs import get_resnet_arch


def get_conv_layer(opt, use_3D=False):
    if "spectral" in opt.norm_G:
        if use_3D:
            conv_layer_base = lambda in_c, out_c, k, p, s: nn.utils.spectral_norm(
                nn.Conv3d(in_c, out_c, k, p, s)
            )
        else:
            conv_layer_base = lambda in_c, out_c, k, p, s: nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, k, p, s)
            )
    else:
        if use_3D:
            conv_layer_base = lambda in_c, out_c, k, p, s: nn.Conv3d(
                in_c, out_c, k, p, s
            )
        else:
            conv_layer_base = lambda in_c, out_c, k, p, s: nn.Conv2d(
                in_c, out_c, k, p, s
            )

    return conv_layer_base


def get_batchnorm_layer(opt):
    norm_G = opt.norm_G.split(":")[1]
    if norm_G == "batch":
        norm_layer = nn.BatchNorm2d
    elif norm_G == "spectral_instance":
        norm_layer = nn.InstanceNorm2d
    elif norm_G == "spectral_batch":
        norm_layer = nn.BatchNorm2d
    elif norm_G == "spectral_batchstanding":
        norm_layer = BatchNorm_StandingStats

    return norm_layer


# VGG architecture, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Normalize the image so that it is in the appropriate range
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


#######################################################################################
#       ResNet models
#######################################################################################
class ResNetEncoder(nn.Module):
    """ Modified implementation of the BigGAN model.
    """

    def __init__(
        self,
        opt,
        channels_in=3,
        channels_out=64,
        downsample=True,
        model_type=None,
    ):
        super().__init__()

        if not model_type:
            arch = get_resnet_arch(opt.refine_model_type, opt, channels_in)
        else:
            arch = get_resnet_arch(model_type, opt, channels_in)

        gblocks = []
        for l_id in range(1, len(arch["layers_enc"])):
            gblock = ResNet_Block(
                arch["layers_enc"][l_id - 1],
                arch["layers_enc"][l_id],
                opt,
                (downsample and arch["downsample"][l_id - 1]),
            )
            gblocks += [gblock]

        self.gblocks = nn.Sequential(*gblocks)

    def forward(self, x):
        return self.gblocks(x)


class ResNetDecoder(nn.Module):
    """ Modified implementation of the BigGAN model. """

    def __init__(self, opt, channels_in=64, channels_out=3, use_tanh=True):
        super().__init__()

        arch = get_resnet_arch(opt.refine_model_type, opt)

        eblocks = []
        for l_id in range(1, len(arch["layers_dec"])):
            eblock = ResNet_Block(
                arch["layers_dec"][l_id - 1],
                arch["layers_dec"][l_id],
                opt,
                arch["upsample"][l_id - 1],
            )
            eblocks += [eblock]

        self.eblocks = nn.Sequential(*eblocks)

        if use_tanh:
            self.norm = nn.Tanh()

    def forward(self, x):
        return self.norm(self.eblocks(x))


#######################################################################################
#       UNet models
#######################################################################################


class Unet(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
        self,
        num_filters=32,
        channels_in=3,
        channels_out=3,
        use_tanh=False,
        use_3D=False,
        opt=None,
    ):
        super(Unet, self).__init__()

        conv_layer = get_conv_layer(opt, use_3D=use_3D)

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters * 2, channels_out, 3, 1, 1)

        norm_layer = get_batchnorm_layer(opt)

        self.batch_norm = norm_layer(num_filters)
        self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.batch_norm8_3(self.conv7(self.leaky_relu(e6)))
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e8))))
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(d1))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d5))))
        # state size is (num_filters x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.batch_norm(self.dconv7(self.up(self.relu(d6))))
        # state size is (num_filters) x 128 x 128
        # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.up(self.relu(d7)))
        # state size is (nc) x 256 x 256
        # output = self.tanh(d8)
        return d8


class UNetEncoder(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(self, opt, num_filters=32, channels_in=3, channels_out=3):
        super(UNetEncoder, self).__init__()
        conv_layer = get_conv_layer(opt, use_3D=False)
        norm_layer = get_batchnorm_layer(opt)

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, channels_out, 3, 1, 1)

        self.batch_norm = norm_layer(num_filters)
        self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.batch_norm8_3(self.conv7(self.leaky_relu(e6)))
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e8))))
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(d1))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.dconv5(self.up(self.relu(d4)))
        # state size is (num_filters x 2) x 64 x 64
        return d5_


class UNetEncoder64(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(self, opt, num_filters=32, channels_in=3, channels_out=3):
        super().__init__()
        conv_layer = get_conv_layer(opt, use_3D=False)
        norm_layer = get_batchnorm_layer(opt)

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, channels_out, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2 * 2, channels_out, 3, 1, 1)

        self.batch_norm = norm_layer(num_filters)
        self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.batch_norm8_3(self.conv7(self.leaky_relu(e6)))
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e8))))
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(d1))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.dconv6(self.up(self.relu(d5)))
        # state size is (num_filters x 2) x 64 x 64
        return d6_


class UNetDecoder64(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(self, opt=None, num_filters=32, channels_in=3, channels_out=3):
        super().__init__()
        conv_layer = get_conv_layer(opt, use_3D=False)
        norm_layer = get_batchnorm_layer(opt)

        # self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters, channels_out, 3, 1, 1)

        self.batch_norm = norm_layer(num_filters)
        self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        if opt.normalize_image:
            self.norm = nn.Tanh()
        else:
            self.norm = nn.Sigmoid()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.conv3(input)
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.batch_norm8_3(self.conv7(self.leaky_relu(e6)))
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e8))))
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(d1))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d5))))
        # state size is (num_filters x 2) x 64 x 64
        d7_ = self.batch_norm(self.dconv7(self.up(self.relu(d6_))))
        # state size is (num_filters) x 128 x 128
        # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
        d8 = self.dconv8(self.up(self.relu(d7_)))
        # state size is (nc) x 256 x 256
        # output = self.tanh(d8)
        return self.norm(d8)


class UNetDecoder(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(self, num_filters=32, channels_in=3, channels_out=3, opt=None):
        super(UNetDecoder, self).__init__()
        conv_layer = get_conv_layer(opt, use_3D=False)
        norm_layer = get_batchnorm_layer(opt)

        # self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(channels_in, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters, channels_out, 3, 1, 1)

        self.batch_norm = norm_layer(num_filters)
        self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.batch_norm4_1 = norm_layer(num_filters * 4)
        # self.batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        if opt.normalize_image:
            self.norm = nn.Tanh()
        else:
            self.norm = nn.Sigmoid()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # state size is (num_filters x 2) x 64 x 64
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.conv4(input)
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.batch_norm8_3(self.conv7(self.leaky_relu(e6)))
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e8))))
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(d1))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
        # state size is (num_filters x 4) x 32 x 32
        d6_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d5_))))
        # state size is (num_filters x 2) x 64 x 64
        d7_ = self.batch_norm(self.dconv7(self.up(self.relu(d6_))))
        # state size is (num_filters) x 128 x 128
        # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
        d8 = self.dconv8(self.up(self.relu(d7_)))
        # state size is (nc) x 256 x 256
        # output = self.tanh(d8)
        return self.norm(d8)


class UNet3D64(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
        self, num_filters=32, channels_in=64, channels_out=64, voxels_size=64
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(channels_in, num_filters, 4, 2, 1)
        self.conv2 = nn.Conv3d(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv3d(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv3d(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = nn.Conv3d(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="trilinear")

        self.dconv2 = nn.Conv3d(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv3 = nn.Conv3d(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv4 = nn.Conv3d(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv5 = nn.Conv3d(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv6 = nn.Conv3d(num_filters * 2, channels_out, 3, 1, 1)

        self.batch_norm2_0 = nn.BatchNorm3d(num_filters * 2)
        self.batch_norm4_0 = nn.BatchNorm3d(num_filters * 4)
        self.batch_norm4_1 = nn.BatchNorm3d(num_filters)
        self.batch_norm8_0 = nn.BatchNorm3d(num_filters * 8)
        self.batch_norm8_1 = nn.BatchNorm3d(num_filters * 8)
        self.batch_norm8_5 = nn.BatchNorm3d(num_filters * 8)
        self.batch_norm8_6 = nn.BatchNorm3d(num_filters * 4)
        self.batch_norm8_7 = nn.BatchNorm3d(num_filters * 2)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 64 x 64
        e1 = self.conv1(input)
        # state size is (num_filters) x 32 x 32
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 16 x 16
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 8 x 8
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 4 x 4
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 2x2

        # Decoder
        # Deconvolution layers:
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(e5))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e4), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e3), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e2), 1)
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e1), 1)
        d6 = self.dconv6(self.up(self.relu(d5)))
        return d6


class UNet3D(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
        self, num_filters=32, channels_in=64, channels_out=64, voxels_size=64
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(channels_in, num_filters, 4, 2, 1)
        self.conv2 = nn.Conv3d(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv3d(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv3d(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = nn.Conv3d(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = nn.Conv3d(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="trilinear")

        self.dconv2 = nn.Conv3d(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv3 = nn.Conv3d(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv4 = nn.Conv3d(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv5 = nn.Conv3d(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv6 = nn.Conv3d(num_filters * 2, channels_out, 3, 1, 1)

        self.batch_norm = nn.BatchNorm3d(num_filters)
        self.batch_norm2_0 = nn.BatchNorm3d(num_filters * 2)
        self.batch_norm4_0 = nn.BatchNorm3d(num_filters * 4)
        self.batch_norm4_1 = nn.BatchNorm3d(num_filters)
        self.batch_norm8_0 = nn.BatchNorm3d(num_filters * 8)
        self.batch_norm8_1 = nn.BatchNorm3d(num_filters * 8)
        self.batch_norm8_2 = nn.BatchNorm3d(num_filters * 8)
        self.batch_norm8_4 = nn.BatchNorm3d(num_filters * 8)
        self.batch_norm8_5 = nn.BatchNorm3d(num_filters * 8)
        self.batch_norm8_6 = nn.BatchNorm3d(num_filters * 4)
        self.batch_norm8_7 = nn.BatchNorm3d(num_filters * 2)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 64 x 64
        e1 = self.conv1(input)
        # state size is (num_filters) x 32 x 32
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 16 x 16
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 8 x 8
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 4 x 4
        e5 = self.conv5(self.leaky_relu(e4))
        # state size is (num_filters x 8) x 8 x 8

        # Decoder
        # Deconvolution layers:
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(e5))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e4), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e3), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e2), 1)
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e1), 1)
        d6 = self.dconv6(self.up(self.relu(d5)))
        return d6
