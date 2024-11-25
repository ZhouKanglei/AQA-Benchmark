#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/25 15:45:20

import torch
import pickle
import numpy as np


class LOGO(torch.utils.data.Dataset):
    def __init__(self, args, subset=None):

        self.args = args
        self.phase = args.phase if subset is None else subset
        args.logger.info(f"- {self.phase} dataset loading")

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

    def __getitem__(self, index):
        sample = self.dataset[index]

        data = {}
        data["video"] = self.load_video(sample, self.phase)
        data["score"] = self.label_dict[sample][1]

        return data

    def __len__(self):
        if self.args.debug:
            return 64
        else:
            return len(self.dataset)
