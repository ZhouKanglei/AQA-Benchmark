#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/09 11:22:18

import csv
import os
from pydoc import locate

import numpy as np
import torch
import yaml

from thop import profile

from utils.logger import Logger
from utils.losses import Criterion
from utils.misc import copy_dir, count_param, init_seed, worker_init_fn


class Processor(object):
    """
    Processor for data processing.
    """

    def __init__(self, args):
        """
        Init processor.
        """
        self.args = args

        # initialize with args
        self.init()
        # initialize logger
        self.init_logger()
        if self.args.phase == "train":
            # backup the codes
            self.backup()
            # save the config
            self.save_config()
        # initialize seed
        self.init_seed()

        # build criterion
        self.build_criterion()
        # build dataloader
        self.build_dataloader()

        # build model
        self.build_model()
        # model to device
        self.model2gpu()
        # build optimizer and scheduler
        self.build_opt_sch()

    def init(self):
        """
        Initialize.
        """
        # get the experiment name: backbone + neck + head
        self.exp_name = ""
        for model in ["backbone", "neck", "head"]:
            if hasattr(self.args, model):
                if getattr(self.args, model) is not None:
                    if self.exp_name != "":
                        self.exp_name += "-"
                    self.exp_name += getattr(self.args, model)
                else:
                    if self.exp_name != "":
                        self.exp_name += "-"
                    self.exp_name += "none"

            ## get the model args
            args_str = ""
            if hasattr(self.args, model + "_args"):
                if getattr(self.args, model + "_args") is not None:
                    for k, v in getattr(self.args, model + "_args").items():
                        if args_str != "":
                            args_str += "_"
                        args_str += f"{k[0]}={v}"

            if "backbone" not in model and args_str != "":
                self.exp_name += f"'{args_str}"

        if self.args.exp_name is not None:
            self.exp_name += "-" + self.args.exp_name
        else:
            self.exp_name += f"-{self.args.timestamp}"

        # work directory: outputs + dataset + experiment name
        self.config_name = os.path.basename(self.args.config).split(".")[0]
        self.work_dir = f"{self.args.output_dir}/{self.config_name}/{self.exp_name}"
        os.makedirs(self.work_dir, exist_ok=True)

        # weight save directory
        self.weight_save_dir = os.path.join(self.work_dir, "weights")
        os.makedirs(self.weight_save_dir, exist_ok=True)

        # results save directory
        self.results_save_dir = os.path.join(self.args.output_dir, "res")
        os.makedirs(self.results_save_dir, exist_ok=True)

        # gpus to list
        if hasattr(self.args, "gpus_non_default"):
            self.args.gpus = [int(gpu) for sublist in self.args.gpus for gpu in sublist]
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in self.args.gpus])
        # self.args.gpus = list(range(len(self.args.gpus)))
        self.device = torch.device(f"cuda:{self.args.gpus[0]}")

    def init_logger(self):
        """
        Initialize logger.
        """
        # log file
        self.log_dir = f"{self.work_dir}/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir, f"{self.args.phase}-{self.args.timestamp}.log"
        )

        self.logger = Logger(self.log_file, self.exp_name)
        self.args.logger = self.logger

        self.logger.info(f"Initilization - {self.exp_name}", level=0)
        self.logger.info(f"Timestamp: {self.args.timestamp}")
        self.logger.info(f" Exp name: {self.exp_name}")
        self.logger.info(f" Work dir: {self.work_dir}")
        self.logger.info(f" Log file: {self.log_file}")

    def backup(self):
        """
        Backup the codes.
        """
        self.logger.info("Backup the codes:", level=1)
        # save main file and dirs
        for dir in ["main.py", "utils", "datasets", "models"]:
            new_dir = os.path.join(self.work_dir, dir)
            if os.path.exists(dir):
                copy_dir(dir, new_dir)
                self.logger.info(f"- Backup {dir:8} to {new_dir}")

    def save_config(self):
        """
        Save the config.
        """
        # save the config
        self.config_dir = os.path.join(self.work_dir, "configs")
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_file = os.path.join(
            self.config_dir, f"{self.args.phase}-{self.args.timestamp}.yaml"
        )
        with open(self.config_file, "w") as f:
            yaml.dump(vars(self.args), f, default_flow_style=False)
        self.logger.info(f"- Backup {'config':8} to {self.config_file}")

    def init_seed(self):
        """
        Initialize seed.
        """
        init_seed(self.args.seed)
        self.logger.info("Initialize seed:", level=1)
        self.logger.info(f"- Seed: {self.args.seed}")
        self.logger.info(f"- Seed for torch and numpy.")
        self.logger.info(f"- Set torch seed.")
        self.logger.info(f"- Set torch cuda seed.")
        self.logger.info(f"- Set torch cudnn deterministic.")
        self.logger.info(f"- Set torch cudnn benchmark.")

    def build_model(self):
        """
        Build model.
        """

        self.logger.info("Build model:", level=1)
        self.logger.info(
            f"- Backbone: {self.args.backbone:6s} with args {self.args.backbone_args}"
        )
        if self.args.neck is not None:
            self.logger.info(
                f"-     Neck: {self.args.neck:6s} with args {self.args.neck_args}"
            )
        self.logger.info(
            f"-     Head: {self.args.head:6s} with args {self.args.head_args}"
        )

        self.args.backbone_args["logger"] = self.logger
        model = locate("models." + self.args.model)
        self.model = model(self.args)

        total_params = count_param(self.model)
        back_params = count_param(self.model.backbone)
        neck_params = count_param(self.model.neck)
        head_params = count_param(self.model.head)
        self.logger.info(
            f"- Total params: {total_params:,d} = {back_params:,d} + {neck_params:,d} + {head_params:,d}"
        )

    def model2gpu(self):
        """
        Model to GPU.
        """
        self.logger.info("Model to GPU:", level=1)
        self.logger.info(f"- GPUs: {self.args.gpus}")

        if len(self.args.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpus)
            self.model = self.model.cuda(self.args.gpus[0])
            self.logger.info(f"- DataParallel with GPUs: {self.args.gpus}")
        else:
            self.model = self.model.to(self.device)
            self.logger.info(f"- Model to GPU: {self.device}")

    def build_dataloader(self):
        """
        Build dataloader.
        """
        self.logger.info("Build dataloader:", level=1)
        self.logger.info(f"- Dataset: {self.args.dataset}")

        dataset = locate("datasets." + self.args.dataset)
        self.dataset = dataset(self.args)
        if hasattr(self.dataset, "delta"):
            self.logger.info("- Dataset has delta function.")
            self.args.train_deltas = self.dataset.delta()

        # write a train data loader for the dataset
        if not hasattr(self.args, "train_batch_size"):
            self.args.train_batch_size = self.args.train_batch_size_per_gpu * len(
                self.args.gpus
            )
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        # write a test data loader for the dataset
        if not hasattr(self.args, "test_batch_size"):
            self.args.test_batch_size = self.args.test_batch_size_per_gpu * len(
                self.args.gpus
            )
        self.dataset = dataset(self.args, subset="test")
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        self.logger.info(f"- Train batch size: {self.args.train_batch_size}")
        self.logger.info(f"-  Test batch size: {self.args.test_batch_size}")
        self.logger.info(f"-     # of workers: {self.args.num_workers}")

    def build_opt_sch(self):
        """
        Build optimizer and scheduler.
        """
        self.logger.info("Build optimizer and scheduler:", level=1)
        self.logger.info(f"- Optimizer: {self.args.optimizer}")
        self.logger.info(f"-   Base lr: {self.args.base_lr}")
        self.logger.info(f"- Lr factor: {self.args.lr_factor}")
        self.logger.info(f"- We. decay: {self.args.weight_decay}")

        self.logger.info(f"- Scheduler: {self.args.scheduler}")

        if self.args.optimizer == "adam":
            if hasattr(self.model, "module"):
                model = self.model.module
            else:
                model = self.model
            self.optimizer = torch.optim.Adam(
                [
                    {
                        "params": model.backbone.parameters(),
                        "lr": self.args.base_lr * self.args.lr_factor,
                    },
                    {"params": model.neck.parameters()},
                    {"params": model.head.parameters()},
                ],
                lr=self.args.base_lr,
                weight_decay=int(self.args.weight_decay),
            )
        else:
            raise NotImplementedError()

        self.scheduler = None

    def build_criterion(self):
        """
        Build criterion.
        """
        self.logger.info("Build criterion:", level=1)
        self.logger.info(f"- Criterion: {self.args.criterion}")

        self.criterion = Criterion(self.args)

    def save_model_weight(self, weight_name="w", dict=None):
        """
        Save model weight.
        """
        model_file = os.path.join(self.weight_save_dir, f"{weight_name}.pth")
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if dict is not None:
            save_dict.update(dict)
        torch.save(save_dict, model_file)
        self.logger.info(f"Save weight to {model_file}")

    def load_model_weight(self, weight_name="w", weight_path=None):
        """
        Load model weight.
        """
        self.logger.info("Load model weight:", level=1)
        model_file = os.path.join(self.weight_save_dir, f"{weight_name}.pth")

        # if weight_path is not None, load the weight from the weight_path
        if weight_path is not None:
            model_file = weight_path
        if not os.path.exists(model_file):
            self.logger.info(f"- No model weight file {model_file}")
            return

        save_dict = torch.load(model_file, map_location=self.device)
        # if model dict is on multiple gpus, and current model is on single gpu
        if "module" in list(save_dict["model"].keys())[0] and not hasattr(
            self.model, "module"
        ):
            new_model_dict = {}
            for k, v in save_dict["model"].items():
                new_model_dict[k[7:]] = v
            save_dict["model"] = new_model_dict
        # if model dict is on single gpu, and current model is on multiple gpus
        if not "module" in list(save_dict["model"].keys())[0] and hasattr(
            self.model, "module"
        ):
            new_model_dict = {}
            for k, v in save_dict["model"].items():
                new_model_dict["module." + k] = v
            save_dict["model"] = new_model_dict
        # load model weight
        self.model.load_state_dict(save_dict["model"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        # print the other information in the save_dict
        self.logger.info(f"- Load weight from {model_file}")
        for k, v in save_dict.items():
            if k not in ["model", "optimizer"]:
                setattr(self, k, v)
                self.logger.info(
                    f"- {k:>15s}: {v:.4f}"
                    if isinstance(v, np.float64)
                    else f"- {k:>15s}: {v}"
                )

    def data2device(self, batch_data):
        # Convert lists to tensors if necessary
        batch_data = {
            k: (
                [item.to(self.device).to(torch.float32) for item in v]
                if isinstance(v, list)
                else v.to(self.device).to(torch.float32)
            )
            for k, v in batch_data.items()
        }

        return batch_data

    def measure_model(self, batch_data):
        self.logger.info("Model performance:", level=1)
        # get single data from a batch data
        dummy_input = {
            k: ([item[:1] for item in v] if isinstance(v, list) else v[:1])
            for k, v in batch_data.items()
        }
        dummy_clip_input = dummy_input["video"][:, :, :16]
        # MEASURE FLOPS AND PARAMS for backbone
        model = self.model.module if hasattr(self.model, "module") else self.model
        b_macs, b_params = profile(
            model.backbone, inputs=(dummy_clip_input,), verbose=0
        )

        self.b_flops = b_macs * 2 / 1e9 * 10  # GFlops for 10 clips
        self.b_params = b_params / 1e6

        self.logger.info(f"- Backbone Flops: {self.b_flops:10.6f} G")
        self.logger.info(f"-         Params: {self.b_params:10.6f} M")

        # MEASURE FLOPS AND PARAMS for the entire model
        macs, params = profile(self.model, inputs=(dummy_input,), verbose=0)
        self.flops = macs * 2 / 1e9  # GFlops
        self.params = params / 1e6  # MParams

        self.logger.info(f"-      All Flops: {self.flops:10.6f} G")
        self.logger.info(f"-         Params: {self.params:10.6f} M")

        # MEASURE PERFORMANCE
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )

        repetitions = 10
        timings = np.zeros((repetitions, 1))

        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = self.model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        self.infer_time = timings.mean() / 1000  # seconds
        self.infer_time_std = timings.std() / 1000  # seconds

        self.logger.info(f"- Inference time: {self.infer_time:10.6f} s")
        self.logger.info(f"- Infer time std: {self.infer_time_std:10.6f} s")

        self.save_inference_result()

    def save_inference_result(self):
        """
        write the performance to a file
        """
        results_path = os.path.join(self.results_save_dir, "stat_inference.csv")
        if not os.path.exists(results_path):
            with open(results_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "timestamp",
                        "flops",
                        "params",
                        "backbone_flops",
                        "backbone_params",
                        "infer_time",
                        "infer_time_std",
                        "config_name",
                        "exp_name",
                    ]
                )

            self.logger.info(f"Create results file {results_path}", level=1)

        with open(results_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    self.args.timestamp,
                    f"{self.flops:.6f}",
                    f"{self.params:.6f}",
                    f"{self.b_flops:.6f}",
                    f"{self.b_params:.6f}",
                    f"{self.infer_time:.6f}",
                    f"{self.infer_time_std:.6f}",
                    self.config_name,
                    self.exp_name,
                ]
            )

            self.logger.info(f"Results saved to {results_path}", level=1)

    def save_best_result(self, evaluator):
        """
        Save testing results into a csv file.
        timestamp, best_srcc_epoch, best_srcc, L2, RL2, exp name
        srcc, l2, rl2e maintain 4 decimal places.
        """
        results_path = os.path.join(self.results_save_dir, "stat_best_res.csv")

        if not os.path.exists(results_path):
            with open(results_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "timestamp",
                        "best_srcc_epoch",
                        "best_srcc",
                        "L2",
                        "RL2",
                        "config_name",
                        "exp_name",
                    ]
                )

            self.logger.info(f"Create results file {results_path}", level=1)

        with open(results_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    self.args.timestamp,
                    f"{evaluator.best_srcc:.4f}",
                    f"{evaluator.l2:.4f}",
                    f"{evaluator.rl2e:.4f}",
                    -1 if hasattr(self, "best_srcc_epoch") else self.best_srcc_epoch,
                    self.config_name,
                    self.exp_name,
                ]
            )

            self.logger.info(f"Results saved to {results_path}", level=1)

    def process(self):
        """
        Process data.
        """
        pass
