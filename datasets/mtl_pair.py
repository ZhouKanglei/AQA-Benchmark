#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/09 17:52:26

import torch
import os
import pickle
import random
import glob

from PIL import Image
from utils.misc import get_video_trans


class MTL_PAIR(torch.utils.data.Dataset):

    def __init__(self, args, subset="train"):
        self.args = args
        self.phase = args.phase if subset is None else subset
        args.logger.info(f"- {self.phase} dataset loading")

        self.transforms = get_video_trans(self.phase)
        # using Difficult Degree
        self.usingDD = args.usingDD
        # some flags
        self.dive_number_choosing = args.dive_number_choosing
        # file path
        self.label_path = args.label_path
        self.split_path = args.train_split
        self.split = self.read_pickle(self.split_path)
        self.label_dict = self.read_pickle(self.label_path)
        self.data_root = args.data_root
        # setting
        self.temporal_shift = args.temporal_shift
        self.voter_number = args.voter_number
        self.length = args.frame_length
        # build difficulty dict (difficulty of each action, the cue to choose exemplar)
        self.difficulties_dict = {}
        self.dive_number_dict = {}

        if self.phase == "test":
            self.split_path_test = args.test_split
            self.split_test = self.read_pickle(self.split_path_test)
            self.difficulties_dict_test = {}
            self.dive_number_dict_test = {}

        if self.usingDD:
            self.preprocess()
            self.check()

        self.choose_list = self.split.copy()
        if self.phase == "test":
            self.dataset = self.split_test
        else:
            self.dataset = self.split

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

    def preprocess(self):
        if self.dive_number_choosing:
            # Dive Number
            for item in self.split:
                dive_number = self.label_dict.get(item)["dive_number"]
                if self.dive_number_dict.get(dive_number) is None:
                    self.dive_number_dict[dive_number] = []
                self.dive_number_dict[dive_number].append(item)

            if self.phase == "test":
                for item in self.split_test:
                    dive_number = self.label_dict.get(item)["dive_number"]
                    if self.dive_number_dict_test.get(dive_number) is None:
                        self.dive_number_dict_test[dive_number] = []
                    self.dive_number_dict_test[dive_number].append(item)
        else:
            # DD
            for item in self.split:
                difficulty = self.label_dict.get(item)["difficulty"]
                if self.difficulties_dict.get(difficulty) is None:
                    self.difficulties_dict[difficulty] = []
                self.difficulties_dict[difficulty].append(item)

            if self.phase == "test":
                for item in self.split_test:
                    difficulty = self.label_dict.get(item)["difficulty"]
                    if self.difficulties_dict_test.get(difficulty) is None:
                        self.difficulties_dict_test[difficulty] = []
                    self.difficulties_dict_test[difficulty].append(item)

    def check(self):
        if self.dive_number_choosing:
            # dive_number_dict
            for key in sorted(list(self.dive_number_dict.keys())):
                file_list = self.dive_number_dict[key]
                for item in file_list:
                    assert self.label_dict[item]["dive_number"] == key

            if self.phase == "test":
                for key in sorted(list(self.dive_number_dict_test.keys())):
                    file_list = self.dive_number_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item]["dive_number"] == key
        else:
            # difficulties_dict
            for key in sorted(list(self.difficulties_dict.keys())):
                file_list = self.difficulties_dict[key]
                for item in file_list:
                    assert self.label_dict[item]["difficulty"] == key

            if self.phase == "test":
                for key in sorted(list(self.difficulties_dict_test.keys())):
                    file_list = self.difficulties_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item]["difficulty"] == key

        self.args.logger.info("- Check done")

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
                                    self.label_dict[file_list[i]]["final_score"]
                                    / self.label_dict[file_list[i]]["difficulty"]
                                    - self.label_dict[file_list[j]]["final_score"]
                                    / self.label_dict[file_list[j]]["difficulty"]
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
                                    self.label_dict[file_list[i]]["final_score"]
                                    / self.label_dict[file_list[i]]["difficulty"]
                                    - self.label_dict[file_list[j]]["final_score"]
                                    / self.label_dict[file_list[j]]["difficulty"]
                                )
                            )
        else:
            delta = []
            dataset = self.split.copy()
            for i in range(len(dataset)):
                for j in range(i + 1, len(dataset)):
                    delta.append(
                        abs(
                            self.label_dict[dataset[i]]["final_score"]
                            - self.label_dict[dataset[j]]["final_score"]
                        )
                    )

        return delta

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        data = {}
        if self.phase == "test":
            # test phase
            data["video"] = self.load_video(sample_1, "test")
            data["score"] = self.label_dict.get(sample_1).get("final_score")
            data["difficulty"] = self.label_dict.get(sample_1).get("difficulty")
            data["completeness"] = data["score"] / data["difficulty"]

            if self.usingDD:
                # NOTE: using Dive Number to choose
                if self.dive_number_choosing:
                    train_file_list = self.dive_number_dict[
                        self.label_dict[sample_1]["dive_number"]
                    ]
                    random.shuffle(train_file_list)
                    choosen_sample_list = train_file_list[: self.voter_number]
                else:
                    # choose a list of sample in training_set
                    train_file_list = self.difficulties_dict[
                        self.label_dict[sample_1]["difficulty"]
                    ]
                    random.shuffle(train_file_list)
                    choosen_sample_list = train_file_list[: self.voter_number]
            else:
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[: self.voter_number]
                
            # if choosen_sample_list is less than the number of voters, then repeat choosing 
            while len(choosen_sample_list) < self.voter_number:
                random.shuffle(train_file_list)
                choosen_sample_list += train_file_list[: self.voter_number - len(choosen_sample_list)]

            data["target_video"] = []
            data["target_score"] = []
            data["target_difficulty"] = []
            data["target_completeness"] = []

            for item in choosen_sample_list:
                data["target_video"].append(self.load_video(item, "train"))
                data["target_score"].append(
                    self.label_dict.get(item).get("final_score")
                )
                data["target_difficulty"].append(
                    self.label_dict.get(item).get("difficulty")
                )
                data["target_completeness"].append(
                    data["target_score"][-1] / data["target_difficulty"][-1]
                )

            return data
        else:
            # train phase
            data["video"] = self.load_video(sample_1, "train")
            data["score"] = self.label_dict.get(sample_1).get("final_score")
            data["difficulty"] = self.label_dict.get(sample_1).get("difficulty")
            data["completeness"] = data["score"] / data["difficulty"]

            # choose a sample
            if self.usingDD:
                # did not using a pytorch sampler, using diff_dict to pick a video sample
                if self.dive_number_choosing:
                    # NOTE: using Dive Number to choose
                    file_list = self.dive_number_dict[
                        self.label_dict[sample_1]["dive_number"]
                    ].copy()
                else:
                    # all sample owning same difficulties
                    file_list = self.difficulties_dict[
                        self.label_dict[sample_1]["difficulty"]
                    ].copy()
            else:
                # randomly
                file_list = self.split.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            # sample 2
            data["target_video"] = self.load_video(sample_2, "train")
            data["target_score"] = self.label_dict.get(sample_2).get("final_score")
            data["target_difficulty"] = self.label_dict.get(sample_2).get("difficulty")
            data["target_completeness"] = (
                data["target_score"] / data["target_difficulty"]
            )
            return data

    def __len__(self):
        if self.args.debug:
            return 21
        else:
            return len(self.dataset)
