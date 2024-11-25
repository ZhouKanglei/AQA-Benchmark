#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/25 15:58:24

import torch
import pickle
import numpy as np
import random


class LOGO_PAIR(torch.utils.data.Dataset):
    def __init__(self, args, subset=None):
        random.seed(args.seed)
        self.args = args
        self.phase = args.phase if subset is None else subset
        args.logger.info(f"- {self.phase} dataset loading")

        self.voter_number = args.voter_number

        # file path
        self.data_root = args.data_root
        self.label_path = args.label_path
        self.label_dict = self.read_pickle(self.label_path)

        self.split_path = args.test_split if self.phase == "test" else args.train_split
        self.dataset = self.read_pickle(self.split_path)

    def load_video(self, sample, phase):
        video = np.load(
            f"{self.data_root}/{sample[0]}_{sample[1]}.npy", allow_pickle=True
        )
        return video

    def read_pickle(self, pickle_path):
        with open(pickle_path, "rb") as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def delta(self):
        delta = []
        dataset = self.dataset.copy()
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                delta.append(
                    abs(self.label_dict[dataset[i]][1] - self.label_dict[dataset[j]][1])
                )
        return delta

    def __getitem__(self, index):
        sample = self.dataset[index]

        data = {}
        data["video"] = self.load_video(sample, self.phase)
        data["score"] = self.label_dict[sample][1]

        if self.phase == "test":
            # choose a list of sample in training_set
            train_file_list = self.dataset.copy()
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
            for tmp_idx in choosen_sample_list:

                video = self.load_video(tmp_idx, self.phase)
                score = self.label_dict[tmp_idx][1]

                data["target_video"].append(video)
                data["target_score"].append(score)

        else:

            # choose a sample
            # did not using a pytorch sampler, using diff_dict to pick a video sample
            file_list = self.dataset.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample))

            # choosing one out
            tmp_idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[tmp_idx]

            # sample 2
            data["target_video"] = self.load_video(sample_2, self.phase)
            data["target_score"] = self.label_dict[sample_2][1]

        return data

    def __len__(self):
        if self.args.debug:
            return 64
        else:
            return len(self.dataset)
