#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/10 15:12:01

import torch
import os
import pickle
import random
import glob

from PIL import Image

from utils.misc import get_video_trans


class MTL(torch.utils.data.Dataset):
    def __init__(self, args, subset=None):
        random.seed(args.seed)
        self.args = args
        self.phase = args.phase if subset is None else subset
        args.logger.info(f"- {self.phase} dataset loading")

        self.transforms = get_video_trans(self.phase)
        # file path
        self.data_root = args.data_root
        self.label_path = args.label_path
        self.label_dict = self.read_pickle(self.label_path)

        # setting
        self.temporal_shift = args.temporal_shift
        self.length = args.frame_length

        self.split_path = args.test_split if self.phase == "test" else args.train_split
        self.dataset = self.read_pickle(self.split_path)

    def load_video(self, video_file_name, phase):
        image_list = sorted(
            (
                glob.glob(
                    os.path.join(
                        self.data_root,
                        str("{:02d}".format(video_file_name[0])),
                        "*.jpg",
                    )
                )
            )
        )
        end_frame = self.label_dict.get(video_file_name).get("end_frame")
        if phase == "train":
            temporal_aug_shift = random.randint(
                self.temporal_shift[0], self.temporal_shift[1]
            )
            end_frame = end_frame + temporal_aug_shift
        start_frame = end_frame - self.length

        video = [Image.open(image_list[start_frame + i]) for i in range(self.length)]
        return self.transforms(video)

    def read_pickle(self, pickle_path):
        with open(pickle_path, "rb") as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __getitem__(self, index):
        sample = self.dataset[index]

        data = {}
        data["video"] = self.load_video(sample, self.phase)
        data["score"] = self.label_dict.get(sample).get("final_score")
        data["difficulty"] = self.label_dict.get(sample).get("difficulty")
        data["completeness"] = data["score"] / data["difficulty"]
        data["judge_scores"] = self.label_dict.get(sample).get("judge_scores")

        return data

    def __len__(self):
        if self.args.debug:
            return 64
        else:
            return len(self.dataset)
