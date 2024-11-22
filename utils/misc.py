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


def get_video_trans(phase="train", resolution=224):

    if resolution == 224:
        size = (455, 256)
    elif resolution == 112:
        size = (227, 128)
    else:
        raise ValueError(f"Resolution {resolution} not supported")

    train_trans = video_transforms.Compose(
        [
            video_transforms.RandomHorizontalFlip(),
            video_transforms.Resize(size),
            video_transforms.RandomRotation(10),
            video_transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            video_transforms.RandomCrop(resolution),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    test_trans = video_transforms.Compose(
        [
            video_transforms.Resize(size),
            video_transforms.CenterCrop(resolution),
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


def denormalize(label, class_idx, upper=100.0):
    label_ranges = {
        1: (21.6, 102.6),
        2: (12.3, 16.87),
        3: (8.0, 50.0),
        4: (8.0, 50.0),
        5: (46.2, 104.88),
        6: (49.8, 99.36),
    }
    label_range = label_ranges[class_idx]

    true_label = (label / upper) * (label_range[1] - label_range[0]) + label_range[0]
    return true_label


def normalize(label, class_idx, upper=100.0):
    label_ranges = {
        1: (21.6, 102.6),
        2: (12.3, 16.87),
        3: (8.0, 50.0),
        4: (8.0, 50.0),
        5: (46.2, 104.88),
        6: (49.8, 99.36),
    }
    label_range = label_ranges[class_idx]

    norm_label = ((label - label_range[0]) / (label_range[1] - label_range[0])) * upper
    return norm_label
