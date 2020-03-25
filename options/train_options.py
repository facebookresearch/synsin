# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import datetime
import os
import time


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="self-supervised view synthesis"
        )
        self.add_data_parameters()
        self.add_train_parameters()
        self.add_model_parameters()

    def add_model_parameters(self):
        model_params = self.parser.add_argument_group("model")
        model_params.add_argument(
            "--model_type",
            type=str,
            default="zbuffer_pts",
            choices=(
                "zbuffer_pts",
                "deepvoxels",
                "viewappearance",
                "tatarchenko",
            ),
            help='Model to be used.'
        )
        model_params.add_argument(
            "--refine_model_type", type=str, default="unet",
            help="Model to be used for the refinement network and the feature encoder."
        )
        model_params.add_argument(
            "--accumulation",
            type=str,
            default="wsum",
            choices=("wsum", "wsumnorm", "alphacomposite"),
            help="Method for accumulating points in the z-buffer. Three choices: wsum (weighted sum), wsumnorm (normalised weighted sum), alpha composite (alpha compositing)"
        )

        model_params.add_argument(
            "--depth_predictor_type",
            type=str,
            default="unet",
            choices=("unet", "hourglass", "true_hourglass"),
            help='Model for predicting depth'
        )
        model_params.add_argument(
            "--splatter",
            type=str,
            default="xyblending",
            choices=("xyblending"),
        )
        model_params.add_argument("--rad_pow", type=int, default=2,
            help='Exponent to raise the radius to when computing distance (default is euclidean, when rad_pow=2). ')
        model_params.add_argument("--num_views", type=int, default=2,
            help='Number of views considered per input image (inlcluding input), we only use num_views=2 (1 target view).')
        model_params.add_argument(
            "--crop_size",
            type=int,
            default=256,
            help="Crop to the width of crop_size (after initially scaling the images to load_size.)",
        )
        model_params.add_argument(
            "--aspect_ratio",
            type=float,
            default=1.0,
            help="The ratio width/height. The final height of the load image will be crop_size/aspect_ratio",
        )
        model_params.add_argument(
            "--norm_D",
            type=str,
            default="spectralinstance",
            help="instance normalization or batch normalization",
        )
        model_params.add_argument(
            "--noise", type=str, default="", choices=("style", "")
        )
        model_params.add_argument(
            "--learn_default_feature", action="store_true", default=True
        )
        model_params.add_argument(
            "--use_camera", action="store_true", default=False
        )

        model_params.add_argument("--pp_pixel", type=int, default=128,
            help='K: the number of points to conisder in the z-buffer.'
        )
        model_params.add_argument("--tau", type=float, default=1.0,
            help='gamma: the power to raise the distance to.'
        )
        model_params.add_argument(
            "--use_gt_depth", action="store_true", default=False
        )
        model_params.add_argument(
            "--train_depth", action="store_true", default=False
        )
        model_params.add_argument(
            "--only_high_res", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_inverse_depth", action="store_true", default=False,
            help='If true the depth is sampled as a long tail distribution, else the depth is sampled uniformly. Set to true if the dataset has points that are very far away (e.g. a dataset with landscape images, such as KITTI).'
        )
        model_params.add_argument(
            "--ndf",
            type=int,
            default=64,
            help="# of discrim filters in first conv layer",
        )
        model_params.add_argument(
            "--use_xys", action="store_true", default=False
        )
        model_params.add_argument(
            "--output_nc",
            type=int,
            default=3,
            help="# of output image channels",
        )
        model_params.add_argument("--norm_G", type=str, default="batch")
        model_params.add_argument(
            "--ngf",
            type=int,
            default=64,
            help="# of gen filters in first conv layer",
        )
        model_params.add_argument(
            "--radius",
            type=float,
            default=4,
            help="Radius of points to project",
        )
        model_params.add_argument(
            "--voxel_size", type=int, default=64, help="Size of latent voxels"
        )
        model_params.add_argument(
            "--num_upsampling_layers",
            choices=("normal", "more", "most"),
            default="normal",
            help="If 'more', adds upsampling layer between the two middle resnet blocks. "
            + "If 'most', also add one more upsampling + resnet layer at the end of the generator",
        )

    def add_data_parameters(self):
        dataset_params = self.parser.add_argument_group("data")
        dataset_params.add_argument("--dataset", type=str, default="mp3d")
        dataset_params.add_argument(
            "--use_semantics", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--config",
            type=str,
            default="/private/home/ow045820/projects/habitat/habitat-api/configs/tasks/pointnav_rgbd.yaml",
        )
        dataset_params.add_argument(
            "--current_episode_train", type=int, default=-1
        )
        dataset_params.add_argument(
            "--current_episode_val", type=int, default=-1
        )
        dataset_params.add_argument("--min_z", type=float, default=0.5)
        dataset_params.add_argument("--max_z", type=float, default=10.0)
        dataset_params.add_argument("--W", type=int, default=256)
        dataset_params.add_argument(
            "--images_before_reset", type=int, default=1000
        )
        dataset_params.add_argument(
            "--image_type",
            type=str,
            default="both",
            choices=(
                "both",
                "translation",
                "rotation",
                "outpaint",
                "fixedRT_baseline",
            ),
        )
        dataset_params.add_argument("--max_angle", type=int, default=45)
        dataset_params.add_argument(
            "--use_z", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_inv_z", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_rgb_features", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_alpha", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--normalize_image", action="store_true", default=False
        )

    def add_train_parameters(self):
        training = self.parser.add_argument_group("training")
        training.add_argument("--num_workers", type=int, default=0)
        training.add_argument("--start-epoch", type=int, default=0)
        training.add_argument("--num-accumulations", type=int, default=1)
        training.add_argument("--lr", type=float, default=1e-3)
        training.add_argument("--lr_d", type=float, default=1e-3 * 2)
        training.add_argument("--lr_g", type=float, default=1e-3 / 2)
        training.add_argument("--momentum", type=float, default=0.9)
        training.add_argument("--beta1", type=float, default=0)
        training.add_argument("--beta2", type=float, default=0.9)
        training.add_argument("--seed", type=int, default=0)
        training.add_argument("--init", type=str, default="")

        training.add_argument(
            "--use_multi_hypothesis", action="store_true", default=False
        )
        training.add_argument("--num_hypothesis", type=int, default=1)
        training.add_argument("--z_dim", type=int, default=128)
        training.add_argument(
            "--netD", type=str, default="multiscale", help="(multiscale)"
        )
        training.add_argument(
            "--niter",
            type=int,
            default=100,
            help="# of iter at starting learning rate. This is NOT the total #epochs."
            + " Total #epochs is niter + niter_decay",
        )
        training.add_argument(
            "--niter_decay",
            type=int,
            default=10,
            help="# of iter at starting learning rate. This is NOT the total #epochs."
            + " Totla #epochs is niter + niter_decay",
        )

        training.add_argument(
            "--losses", type=str, nargs="+", default=['1.0_l1','10.0_content']
        )
        training.add_argument(
            "--discriminator_losses",
            type=str,
            default="pix2pixHD",
            help="(|pix2pixHD|progressive)",
        )
        training.add_argument(
            "--lambda_feat",
            type=float,
            default=10.0,
            help="weight for feature matching loss",
        )
        training.add_argument(
            "--gan_mode", type=str, default="hinge", help="(ls|original|hinge)"
        )

        training.add_argument(
            "--load-old-model", action="store_true", default=False
        )
        training.add_argument(
            "--load-old-depth-model", action="store_true", default=False
        )
        training.add_argument("--old_model", type=str, default="")
        training.add_argument("--old_depth_model", type=str, default="")

        training.add_argument(
            "--no_ganFeat_loss",
            action="store_true",
            help="if specified, do *not* use discriminator feature matching loss",
        )
        training.add_argument(
            "--no_vgg_loss",
            action="store_true",
            help="if specified, do *not* use VGG feature matching loss",
        )
        training.add_argument("--resume", action="store_true", default=False)

        training.add_argument(
            "--log-dir",
            type=str,
            default="/checkpoint/ow045820/logging/viewsynthesis3d/%s/",
        )

        training.add_argument("--batch-size", type=int, default=16)
        training.add_argument("--continue_epoch", type=int, default=0)
        training.add_argument("--max_epoch", type=int, default=500)
        training.add_argument("--folder_to_save", type=str, default="outpaint")
        training.add_argument(
            "--model-epoch-path",
            type=str,
            default="/%s/%s/models/lr%0.5f_bs%d_model%s_spl%s/noise%s_bn%s_ref%s_d%s_"
            + "camxys%s/_init%s_data%s_seed%d/_multi%s_losses%s_i%s_%s_vol_gan%s/",
        )
        training.add_argument(
            "--run-dir",
            type=str,
            default="/%s/%s/runs/lr%0.5f_bs%d_model%s_spl%s/noise%s_bn%s_ref%s_d%s_"
            + "camxys%s/_init%s_data%s_seed%d/_multi%s_losses%s_i%s_%s_vol_gan%s/",
        )
        training.add_argument("--suffix", type=str, default="")
        training.add_argument(
            "--render_ids", type=int, nargs="+", default=[0, 1]
        )
        training.add_argument("--gpu_ids", type=str, default="0")

    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())

        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict = {
                a.dest: getattr(args, a.dest, None)
                for a in group._group_actions
            }
            arg_groups[group.title] = group_dict

        return (args, arg_groups)


def get_timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    st = "2020-03-01"
    return st


def get_log_path(timestamp, opts):
    inputs = "z" + str(opts.use_z) + "_alpha" + str(opts.use_alpha)
    return (
        opts.log_dir % (opts.dataset)
        + "/%s/"
        + opts.run_dir
        % (
            timestamp,
            opts.folder_to_save,
            opts.lr,
            opts.batch_size,
            opts.model_type,
            opts.splatter,
            opts.noise,
            opts.norm_G,
            opts.refine_model_type,
            opts.depth_predictor_type,
            (str(opts.use_camera) + "|" + str(opts.use_xys)),
            opts.init,
            opts.image_type,
            opts.seed,
            str(opts.use_multi_hypothesis),
            "".join(opts.losses).replace("_", "|"),
            inputs,
            opts.suffix,
            opts.discriminator_losses,
        )
    )


def get_model_path(timestamp, opts):
    inputs = "z" + str(opts.use_z) + "_alpha" + str(opts.use_alpha)
    model_path = opts.log_dir % (opts.dataset) + opts.model_epoch_path % (
        timestamp,
        opts.folder_to_save,
        opts.lr,
        opts.batch_size,
        opts.model_type,
        opts.splatter,
        opts.noise,
        opts.norm_G,
        opts.refine_model_type,
        opts.depth_predictor_type,
        (str(opts.use_camera) + "|" + str(opts.use_xys)),
        opts.init,
        opts.image_type,
        opts.seed,
        str(opts.use_multi_hypothesis),
        "".join(opts.losses).replace("_", "|"),
        inputs,
        opts.suffix,
        opts.discriminator_losses,
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return model_path + "/model_epoch.pth"
