# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_eval_parameters()

    def add_eval_parameters(self):
        eval_params = self.parser.add_argument_group("eval")

        eval_params.add_argument("--old_model", type=str, default="")
        eval_params.add_argument("--short_name", type=str, default="")
        eval_params.add_argument("--result_folder", type=str, default="")
        eval_params.add_argument("--test_folder", type=str, default="")
        eval_params.add_argument("--gt_folder", type=str, default="")
        eval_params.add_argument("--batch_size", type=int, default=4)
        eval_params.add_argument("--num_views", type=int, default=2)
        eval_params.add_argument("--num_workers", type=int, default=1)
        eval_params.add_argument(
            "--render_ids", type=int, nargs="+", default=[1]
        )
        eval_params.add_argument(
            "--image_type",
            type=str,
            default="both",
            choices=(
                "both"
            ),
        )
        eval_params.add_argument("--gpu_ids", type=str, default="0")
        eval_params.add_argument("--images_before_reset", type=int, default=100)
        eval_params.add_argument(
            "--test_input_image", action="store_true", default=False
        )
        eval_params.add_argument(
            "--use_videos", action="store_true", default=False
        )
        eval_params.add_argument(
            "--auto_regressive", action="store_true", default=False
        )
        eval_params.add_argument("--use_gt", action="store_true", default=False)
        eval_params.add_argument("--save_data", action="store_true", default=False)
        eval_params.add_argument("--dataset", type=str, default="")
        eval_params.add_argument(
            "--use_higher_res", action="store_true", default=False
        )

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
