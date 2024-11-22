#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/09 11:23:33

import argparse
from datetime import datetime
import yaml


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class cmdAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + "_non_default", True)


class ParseArgs(object):
    """
    Parse arguments.
    """

    def __init__(self):
        """
        Init ParseArgs.
        """

        self.add_args()
        self.merge_config_args()

    def parse(self):
        """
        Parse arguments.
        """
        pass

    def add_args(self):
        """
        Add general arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="config file path",
            action=cmdAction,
        )
        parser.add_argument(
            "--exp_name",
            type=str,
            default=None,
            help="experiment name",
            action=cmdAction,
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="./outputs",
            help="work directory",
            action=cmdAction,
        )
        parser.add_argument(
            "--weight_path",
            type=str,
            default=None,
            help="pretrained weight path",
            action=cmdAction,
        )
        parser.add_argument(
            "--timestamp",
            type=str,
            default=datetime.now().strftime("%y%m%d-%H%M%S"),
            help="timestamp",
            action=cmdAction,
        )
        parser.add_argument(
            "--phase",
            type=str,
            default="train",
            choices=["train", "test"],
            help="train or test",
            action=cmdAction,
        )
        parser.add_argument(
            "--debug",
            type=str2bool,
            default=False,
            help="debug or not",
            action=cmdAction,
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1024,
            help="random seed",
            action=cmdAction,
        )
        parser.add_argument(
            "--gpus",
            type=list,
            nargs="+",
            default=[0, 1],
            help="gpu ids",
            action=cmdAction,
        )
        parser.add_argument(
            "--pre_train",
            type=str2bool,
            default=False,
            help="pre-training from the pre-trained checkpoint",
            action=cmdAction,
        )
        parser.add_argument(
            "--continue_train",
            type=str2bool,
            default=False,
            help="continue training from the saved checkpoint",
            action=cmdAction,
        )
        parser.add_argument(
            "--class_idx",
            type=int,
            default=1,
            help="only for the AQA-7 dataset",
            choices=[1, 2, 3, 4, 5, 6],
            action=cmdAction,
        )

        self.args = parser.parse_args()

    def merge_config_args(self):
        """
        Get arguments from config file.
        """
        with open(self.args.config, "r") as f:
            config_args = yaml.load(f, Loader=yaml.FullLoader)

            # merge config_args with self.args
            for k, v in config_args.items():
                if k not in vars(self.args).keys():
                    setattr(self.args, k, v)
                elif not hasattr(self.args, f"{k}_non_default"):
                    setattr(self.args, k, v)

    def parse(self):
        """
        Parse arguments.
        """
        pass
