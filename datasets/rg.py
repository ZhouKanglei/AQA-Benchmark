#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/22 20:36:07

import torch
import os
import pickle
import numpy as np


class RG(torch.utils.data.Dataset):
    def __init__(self, args, subset=None):

        self.args = args
        self.phase = args.phase if subset is None else subset
        args.logger.info(f"- {self.phase} dataset loading")

        # file path
        self.data_root = args.data_root
        self.clip_num = args.clip_num

        # action type
        classes_name = [
            "Ball",
            "Clubs",
            "Hoop",
            "Ribbon",
        ]
        self.action_type = classes_name[args.class_idx - 1]
        self.split_path = args.test_split if self.phase == "test" else args.train_split
        score_type = args.score_type

        self.labels = self.read_label(self.split_path, score_type, self.action_type)

    def read_label(self, label_path, score_type, action_type):
        fr = open(label_path, "r")
        idx = {"Difficulty_Score": 1, "Execution_Score": 2, "Total_Score": 3}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()

            if action_type == "all" or action_type == line[0].split("_")[0]:
                labels.append([line[0], float(line[idx[score_type]])])
            if action_type == "TES":
                labels.append([line[0], float(line[1])])
            if action_type == "PCS":
                labels.append([line[0], float(line[2])])
        return labels

    def read_pickle(self, pickle_path):
        with open(pickle_path, "rb") as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def load_video(self, idx, phase):
        video_feat = np.load(os.path.join(self.data_root, self.labels[idx][0] + ".npy"))
        # temporal random crop or padding
        if phase == "train":
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st : st + self.clip_num]

            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[: video_feat.shape[0]] = video_feat
                video_feat = new_feat
        elif hasattr(self.args, "test_clip_fixed") and self.args.test_clip_fixed:
            if len(video_feat) > self.clip_num:
                st = len(video_feat) // 2 - self.clip_num // 2
                video_feat = video_feat[st : st + self.clip_num]

            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[: video_feat.shape[0]] = video_feat
                video_feat = new_feat

        video_feat = torch.from_numpy(video_feat).float()

        return video_feat

    def __getitem__(self, index):

        data = {}
        data["video"] = self.load_video(index, self.phase)
        data["score"] = self.labels[index][1]
        return data

    def __len__(self):
        if self.args.debug:
            return 64
        else:
            return len(self.labels)
