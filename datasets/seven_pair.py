#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/20 20:41:46

import torch
import scipy.io
import os
import random

from PIL import Image
from utils.misc import get_video_trans, normalize


class SEVEN_PAIR(torch.utils.data.Dataset):
    def __init__(self, args, subset=None):
        random.seed(args.seed)
        self.args = args
        self.phase = args.phase if subset is None else subset
        args.logger.info(f"- {self.phase} dataset loading")

        self.transforms = get_video_trans(self.phase)
        self.temporal_shift = args.temporal_shift
        # file path
        self.data_root = args.data_root

        classes_name = [
            "diving",
            "gym_vault",
            "ski_big_air",
            "snowboard_big_air",
            "sync_diving_3m",
            "sync_diving_10m",
        ]
        self.sport_class = classes_name[args.class_idx - 1]

        self.class_idx = args.class_idx  # sport class index(from 1 begin)
        self.score_range = args.score_range
        # file path
        self.data_root = args.data_root
        self.data_path = os.path.join(
            self.data_root, "data/{}-out".format(self.sport_class)
        )

        # read split
        self.split_path = os.path.join(
            args.train_split if self.phase == "train" else args.test_split
        )
        self.split = scipy.io.loadmat(self.split_path)[
            f"consolidated_{self.phase}_list"
        ]
        self.split = self.split[self.split[:, 0] == self.class_idx].tolist()

        # setting
        self.length = args.frame_length
        self.voter_number = args.voter_number

        self.dataset = self.split.copy()

    def load_video(self, idx):
        video_path = os.path.join(self.data_path, "%03d" % idx)
        img_list = [
            os.path.join(video_path, "img_%05d.jpg" % (i + 1))
            for i in range(self.length)
        ]

        if self.phase == "train":
            temporal_aug_shift = random.randint(
                self.temporal_shift[0], self.temporal_shift[1]
            )

            if temporal_aug_shift > 0:
                # shift right, the start frame is repeated
                img_list = [img_list[0] for i in range(temporal_aug_shift)] + img_list[
                    :-temporal_aug_shift
                ]

            elif temporal_aug_shift < 0:
                # shift left, the end frame is repeated
                img_list = img_list[temporal_aug_shift:] + [
                    img_list[-1] for i in range(temporal_aug_shift)
                ]

        video = [
            Image.open(os.path.join(video_path, "img_%05d.jpg" % (i + 1)))
            for i in range(self.length)
        ]
        return self.transforms(video)

    def delta(self):
        delta = []
        dataset = self.split.copy()
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                delta.append(
                    abs(
                        normalize(dataset[i][2], self.class_idx, self.score_range)
                        - normalize(dataset[j][2], self.class_idx, self.score_range)
                    )
                )
        return delta

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        assert int(sample_1[0]) == self.class_idx
        idx = int(sample_1[1])

        data = {}
        data["video"] = self.load_video(idx)
        data["score"] = normalize(sample_1[2], self.class_idx, self.score_range)

        if self.phase == "test":

            # choose a list of sample in training_set
            train_file_list = self.split.copy()
            random.shuffle(train_file_list)
            choosen_sample_list = train_file_list[: self.voter_number]

            # if choosen_sample_list is less than the number of voters, then repeat choosing
            while len(choosen_sample_list) < self.voter_number:
                random.shuffle(train_file_list)
                choosen_sample_list += train_file_list[
                    : self.voter_number - len(choosen_sample_list)
                ]

            data["target_video"] = []
            data["target_score"] = []
            for item in choosen_sample_list:

                tmp_idx = int(item[1])
                video = self.load_video(tmp_idx)
                score = normalize(item[2], self.class_idx, self.score_range)

                data["target_video"].append(video)
                data["target_score"].append(score)

            return data
        else:

            # choose a sample
            # did not using a pytorch sampler, using diff_dict to pick a video sample
            file_list = self.split.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            tmp_idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[tmp_idx]

            # sample 2
            data["target_video"] = self.load_video(int(sample_2[1]))
            data["target_score"] = normalize(
                sample_2[2], self.class_idx, self.score_range
            )
            return data

    def __len__(self):
        return len(self.dataset)
