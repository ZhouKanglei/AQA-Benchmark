#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/17 11:01:43


import torch
import numpy as np
import os
import pickle
import random
import glob

from PIL import Image
from utils.misc import get_video_trans


class FD(torch.utils.data.Dataset):
    def __init__(self, args, subset="train"):
        # seed
        random.seed(args.seed)
        # init
        self.args = args
        self.phase = args.phase if subset is None else subset
        args.logger.info(f"- {self.phase} dataset loading")

        if hasattr(args, "resolution"):
            self.transforms = get_video_trans(self.phase, resolution=args.resolution)
        else:
            self.transforms = get_video_trans(self.phase)
        # using Difficult Degree
        self.usingDD = args.usingDD
        self.length = args.frame_length

        # file path
        self.data_root = args.data_root
        self.label_dict = self.read_pickle(args.label_path)

        self.split_path = args.test_split if self.phase == "test" else args.train_split
        self.dataset = self.read_pickle(self.split_path)

    def load_video(self, video_file_name):
        image_list = sorted(
            (
                glob.glob(
                    os.path.join(
                        self.data_root,
                        video_file_name[0],
                        str(video_file_name[1]),
                        "*.jpg",
                    )
                )
            )
        )

        start_frame = int(image_list[0].split("/")[-1][:-4])
        end_frame = int(image_list[-1].split("/")[-1][:-4])
        frame_list = np.linspace(start_frame, end_frame, self.length).astype(np.int32)
        image_frame_idx = [frame_list[i] - start_frame for i in range(self.length)]

        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.length)]
        frames_labels = [
            self.label_dict.get(video_file_name)[4][i] for i in image_frame_idx
        ]
        frames_catogeries = list(set(frames_labels))
        frames_catogeries.sort(key=frames_labels.index)
        transitions = [frames_labels.index(c) for c in frames_catogeries]
        
        return (
            self.transforms(video),
            np.array([transitions[1] - 1, transitions[-1] - 1]),
            np.array(frames_labels),
        )

    def read_pickle(self, pickle_path):
        with open(pickle_path, "rb") as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __getitem__(self, index):
        sample = self.dataset[index]

        data = {}
        data["video"], data["transits"], data["frame_labels"] = self.load_video(sample)
        # data["number"] = self.label_dict.get(sample)[0]
        data["score"] = self.label_dict.get(sample)[1]
        data["difficulty"] = self.label_dict.get(sample)[2]
        data["completeness"] = data["score"] / data["difficulty"]

        return data

    def __len__(self):
        if self.args.debug:
            return 21
        else:
            return len(self.dataset)
