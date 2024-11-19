#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/17 11:03:02

import torch
import numpy as np
import os
import pickle
import random
import glob

from PIL import Image
from utils.misc import get_video_trans


class FD_PAIR(torch.utils.data.Dataset):
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
        self.dive_number_choosing = args.dive_number_choosing
        self.length = args.frame_length
        self.voter_number = args.voter_number

        # file path
        self.data_root = args.data_root
        self.label_dict = self.read_pickle(args.label_path)
        with open(args.train_split, "rb") as f:
            self.train_dataset_list = pickle.load(f)
        with open(args.test_split, "rb") as f:
            self.test_dataset_list = pickle.load(f)

        self.dive_number_dict = {}
        self.difficulties_dict = {}
        if self.phase == "train":
            self.dataset = self.train_dataset_list
        else:
            self.dataset = self.test_dataset_list
            self.dive_number_dict_test = {}
            self.difficulties_dict_test = {}

        self.choose_list = self.train_dataset_list.copy()
        if self.dive_number_choosing:
            self.preprocess()
            self.check_exemplar_dict()

    def preprocess(self):
        for item in self.train_dataset_list:
            dive_number = self.label_dict.get(item)[0]
            if self.dive_number_dict.get(dive_number) is None:
                self.dive_number_dict[dive_number] = []
            self.dive_number_dict[dive_number].append(item)
        if self.phase == "test":
            for item in self.test_dataset_list:
                dive_number = self.label_dict.get(item)[0]
                if self.dive_number_dict_test.get(dive_number) is None:
                    self.dive_number_dict_test[dive_number] = []
                self.dive_number_dict_test[dive_number].append(item)

    def check_exemplar_dict(self):
        if self.phase == "train":
            for key in sorted(list(self.dive_number_dict.keys())):
                file_list = self.dive_number_dict[key]
                for item in file_list:
                    assert self.label_dict[item][0] == key
        if self.phase == "test":
            for key in sorted(list(self.dive_number_dict_test.keys())):
                file_list = self.dive_number_dict_test[key]
                for item in file_list:
                    assert self.label_dict[item][0] == key

    def delta(self):
        """
        RT: builder group
        """
        if self.usingDD:
            if self.dive_number_choosing:
                delta = []
                for key in list(self.dive_number_dict.keys()):
                    file_list = self.dive_number_dict[key]
                    for i in range(len(file_list)):
                        for j in range(i + 1, len(file_list)):
                            delta.append(
                                abs(
                                    self.label_dict[file_list[i]][1]
                                    / self.label_dict[file_list[i]][2]
                                    - self.label_dict[file_list[j]][1]
                                    / self.label_dict[file_list[j]][2]
                                )
                            )
            else:
                delta = []
                for key in list(self.difficulties_dict.keys()):
                    file_list = self.difficulties_dict[key]
                    for i in range(len(file_list)):
                        for j in range(i + 1, len(file_list)):
                            delta.append(
                                abs(
                                    self.label_dict[file_list[i]][1]
                                    / self.label_dict[file_list[i]][2]
                                    - self.label_dict[file_list[j]][1]
                                    / self.label_dict[file_list[j]][2]
                                )
                            )
        else:
            delta = []
            dataset = self.split.copy()
            for i in range(len(dataset)):
                for j in range(i + 1, len(dataset)):
                    delta.append(
                        abs(
                            self.label_dict[dataset[i]][1]
                            - self.label_dict[dataset[j]][1]
                        )
                    )

        return delta

    def load_video(self, video_file_name, subset="train"):
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
        sample_1 = self.dataset[index]
        data = {}
        data["video"], data["transits"], data["frame_labels"] = self.load_video(
            sample_1
        )
        # data["number"] = self.label_dict.get(sample_1)[0]
        data["score"] = self.label_dict.get(sample_1)[1]
        data["difficulty"] = self.label_dict.get(sample_1)[2]
        data["completeness"] = data["score"] / data["difficulty"]

        # choose a exemplar
        if self.phase == "train":
            # train phrase
            if self.dive_number_choosing == True:
                file_list = self.dive_number_dict[self.label_dict[sample_1][0]].copy()
            elif self.usingDD == True:
                file_list = self.difficulties_dict[self.label_dict[sample_1][2]].copy()
            else:
                # randomly
                file_list = self.train_dataset_list.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            # sample 2
            (
                data["target_video"],
                data["target_transits"],
                data["target_frame_labels"],
            ) = self.load_video(sample_2, "train")
            data["target_score"] = self.label_dict.get(sample_2)[1]
            data["target_difficulty"] = self.label_dict.get(sample_2)[2]
            data["target_completeness"] = (
                data["target_score"] / data["target_difficulty"]
            )
            return data
        else:
            # test phrase
            if self.dive_number_choosing:
                train_file_list = self.dive_number_dict[self.label_dict[sample_1][0]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[: self.voter_number]
            elif self.usingDD:
                train_file_list = self.difficulties_dict[self.label_dict[sample_1][2]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[: self.voter_number]
            else:
                # randomly
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[: self.voter_number]

            # if choosen_sample_list is less than the number of voters, then repeat choosing
            while len(choosen_sample_list) < self.voter_number:
                random.shuffle(train_file_list)
                choosen_sample_list += train_file_list[
                    : self.voter_number - len(choosen_sample_list)
                ]

            data["target_video"] = []
            data["target_transits"] = []
            data["target_frame_labels"] = []
            data["target_score"] = []
            data["target_difficulty"] = []
            data["target_completeness"] = []

            for item in choosen_sample_list:
                video, target_transits, target_frame_labels = self.load_video(
                    item, "train"
                )
                data["target_video"].append(video)
                data["target_transits"].append(target_transits)
                data["target_frame_labels"].append(target_frame_labels)
                data["target_score"].append(self.label_dict.get(item)[1])
                data["target_difficulty"].append(self.label_dict.get(item)[2])
                data["target_completeness"].append(
                    data["target_score"][-1] / data["target_difficulty"][-1]
                )

            return data

    def __len__(self):
        if self.args.debug:
            return 21
        else:
            return len(self.dataset)
