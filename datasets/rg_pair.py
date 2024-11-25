#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/22 20:36:07

import torch
import os
import pickle
import numpy as np
import random


class RG_PAIR(torch.utils.data.Dataset):
    def __init__(self, args, subset=None):
        random.seed(args.seed)
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
        self.split = list(range(len(self.labels)))

        self.voter_number = args.voter_number

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

    def delta(self):
        delta = []
        dataset = self.split.copy()
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                delta.append(abs(self.labels[i][1] - self.labels[j][1]))
        return delta

    def __getitem__(self, index):

        data = {}
        data["video"] = self.load_video(index, self.phase)
        data["score"] = self.labels[index][1]

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
            for tmp_idx in choosen_sample_list:
                
                video = self.load_video(tmp_idx, self.phase)
                score = self.labels[tmp_idx][1]

                data["target_video"].append(video)
                data["target_score"].append(score)

        else:

            # choose a sample
            # did not using a pytorch sampler, using diff_dict to pick a video sample
            file_list = self.split.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(index))
            # choosing one out
            tmp_idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[tmp_idx]

            # sample 2
            data["target_video"] = self.load_video(sample_2, self.phase)
            data["target_score"] = self.labels[sample_2][1]

        return data

    def __len__(self):
        if self.args.debug:
            return 64
        else:
            return len(self.labels)
