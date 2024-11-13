#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/09 10:37:16

import os
import random
import shutil
import numpy as np
import torch
from torchvideotransforms import video_transforms, volume_transforms


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def print_log(log, logger=None):
    """
    Print log or write it into logger.
    """
    if logger is not None:
        logger.info(log)
    else:
        print(log)


def worker_init_fn(worker_id):
    """
    Init worker in dataloader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def init_seed(seed):
    """
    Set random seed for torch and numpy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def copy_dir(src, dst):
    if os.path.exists(dst) and os.path.isdir(dst):
        shutil.rmtree(dst)

    if os.path.exists(src):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)


def get_video_trans(phase="train"):

    train_trans = video_transforms.Compose(
        [
            video_transforms.RandomHorizontalFlip(),
            video_transforms.Resize((455, 256)),
            video_transforms.RandomCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    test_trans = video_transforms.Compose(
        [
            video_transforms.Resize((455, 256)),
            video_transforms.CenterCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    if phase == "train":
        return train_trans
    else:
        return test_trans


# Compute the PDF using PyTorch
def normal_pdf(x, mean, std):
    var = std**2
    denom = (2 * torch.pi * var) ** 0.5
    num = torch.exp(-((x - mean) ** 2) / (2 * var))
    return num / denom
