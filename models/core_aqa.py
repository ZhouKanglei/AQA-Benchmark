#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/12 17:17:50

import torch
from .aqa import AQA


class Group_helper(object):
    def __init__(self, dataset, depth, Symmetrical=True, Max=None, Min=None):
        """
        dataset : list of deltas (CoRe method) or list of scores (RT method)
        depth : depth of the tree
        Symmetrical: (bool) Whether the group is symmetrical about 0.
                    if symmetrical, dataset only contains th delta bigger than zero.
        Max : maximum score or delta for a certain sports.
        """
        self.dataset = sorted(dataset)  #
        self.length = len(dataset)
        self.num_leaf = 2 ** (depth - 1)
        self.symmetrical = Symmetrical
        self.max = Max
        self.min = Min
        self.Group = [[] for _ in range(self.num_leaf)]
        self.build()

    def build(self):
        """
        separate region of each leaf
        """
        if self.symmetrical:
            # delta in dataset is the part bigger than zero.
            for i in range(self.num_leaf // 2):
                # bulid positive half first
                Region_left = self.dataset[
                    int((i / (self.num_leaf // 2)) * (self.length - 1))
                ]

                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]
                Region_right = self.dataset[
                    int(((i + 1) / (self.num_leaf // 2)) * (self.length - 1))
                ]
                if i == self.num_leaf // 2 - 1:
                    if self.max != None:
                        Region_right = self.max
                    else:
                        Region_right = self.dataset[-1]

                self.Group[self.num_leaf // 2 + i] = [Region_left, Region_right]

            for i in range(self.num_leaf // 2):
                self.Group[i] = [-i for i in self.Group[self.num_leaf - 1 - i]]

            for group in self.Group:
                group.sort()

        else:
            for i in range(self.num_leaf):

                Region_left = self.dataset[int((i / self.num_leaf) * (self.length - 1))]

                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]

                Region_right = self.dataset[
                    int(((i + 1) / self.num_leaf) * (self.length - 1))
                ]

                if i == self.num_leaf - 1:
                    if self.max != None:
                        Region_right = self.max
                    else:
                        Region_right = self.dataset[-1]

                self.Group[i] = [Region_left, Region_right]

    def produce_label(self, scores):
        device = scores.device
        scores = scores.view(-1)

        glabel = torch.zeros((scores.size(0), self.num_leaf), device=device)
        rlabel = torch.full((scores.size(0), self.num_leaf), -1.0, device=device)

        for i in range(self.num_leaf):
            mask_pos = (scores >= self.Group[i][0]) & (scores < self.Group[i][1])
            mask_neg = (scores < self.Group[i][0]) & (scores >= self.Group[i][1])
            mask = mask_pos | mask_neg

            glabel[:, i] = mask.float()

            if self.Group[i][1] == self.Group[i][0]:
                rlabel[mask, i] = scores[mask] - self.Group[i][0]
            else:
                rlabel[mask, i] = (scores[mask] - self.Group[i][0]) / (
                    self.Group[i][1] - self.Group[i][0]
                )

        return glabel, rlabel

    def inference(self, probs, deltas):
        """
        probs: bs * leaf
        delta: bs * leaf
        """
        predictions = torch.zeros(probs.shape[0], device=probs.device)
        for n in range(probs.shape[0]):
            prob = probs[n]
            delta = deltas[n]
            leaf_id = prob.argmax()
            if self.Group[leaf_id][0] == self.Group[leaf_id][1]:
                prediction = self.Group[leaf_id][0] + delta[leaf_id]
            else:
                prediction = (
                    self.Group[leaf_id][0]
                    + (self.Group[leaf_id][1] - self.Group[leaf_id][0]) * delta[leaf_id]
                )
            predictions[n] = prediction
        return predictions.reshape(-1)

    def get_Group(self):
        return self.Group

    def number_leaf(self):
        return self.num_leaf


class CORE_AQA(AQA):
    """
    CORE_AQA model
    """

    def __init__(self, args):
        super(CORE_AQA, self).__init__(args)

        self.score_range = args.score_range
        self.group = Group_helper(
            args.train_deltas,
            args.head_args["depth"],
            Symmetrical=True,
            Max=args.score_range,
            Min=0,
        )

    def get_paired_clip_features(self, target, exemplar, label_target, label_exemplar):
        # spatiotemporal feature
        total_video = torch.cat((target, exemplar), 0)  # 2N C T H W
        total_feature = self.get_clip_features(total_video)  # 2N * N * 1024

        feature_1, feature_2 = torch.chunk(total_feature, 2, dim=0)

        feature_1 = self.neck(feature_1)
        feature_2 = self.neck(feature_2)

        combined_feature_1 = torch.cat(
            (feature_1, feature_2, label_target.unsqueeze(1) / self.score_range), 1
        )  # 1 is exemplar
        combined_feature_2 = torch.cat(
            (feature_2, feature_1, label_exemplar.unsqueeze(1) / self.score_range), 1
        )  # 2 is exemplar
        return combined_feature_1, combined_feature_2

    def forward(self, batch_data):

        video_1 = batch_data["video"]  # N, C, T, H, W
        video_2 = batch_data["target_video"]  # N, C, T, H, W

        if self.usingDD:
            label_1 = batch_data["completeness"]
            label_2 = batch_data["target_completeness"]
        else:
            label_1 = batch_data["score"]
            label_2 = batch_data["target_score"]

        if self.training:
            # forward
            com_feat_1, com_feat_2 = self.get_paired_clip_features(
                video_1, video_2, label_1, label_2
            )

            com_feat = torch.cat((com_feat_1, com_feat_2), 0)
            out_prob, delta = self.head(com_feat)

            # tree-level label
            glabel_1, rlabel_1 = self.group.produce_label(label_2 - label_1)
            glabel_2, rlabel_2 = self.group.produce_label(label_1 - label_2)

            # predictions
            leaf_probs = out_prob[-1].reshape(com_feat.shape[0], -1)
            leaf_probs_1, leaf_probs_2 = torch.chunk(leaf_probs, 2, dim=0)
            delta_1, delta_2 = torch.chunk(delta, 2, dim=0)

            # evaluate result of training phase
            relative_scores = self.group.inference(leaf_probs_2, delta_2)

            pred_score = relative_scores + label_2

            if self.usingDD:
                pred_score = pred_score * batch_data["difficulty"]
 
            outputs = {
                "preds": pred_score,
                "labels": batch_data["score"],
                "loss_preds": pred_score,
                "loss_labels": label_1,
                "leaf_probs_1": leaf_probs_1,
                "leaf_probs_2": leaf_probs_2,
                "glabel_1": glabel_1,
                "glabel_2": glabel_2,
                "rlabel_1": rlabel_1,
                "rlabel_2": rlabel_2,
                "delta_1": delta_1,
                "delta_2": delta_2,
            }

        else:
            avg_pred_score = 0
            for v2, l2 in zip(video_2, label_2):
                pred_score = 0
                _, com_feat = self.get_paired_clip_features(video_1, v2, label_1, l2)

                out_prob, delta = self.head(com_feat)

                leaf_probs = out_prob[-1].reshape(com_feat.shape[0], -1)
                delta = delta.reshape(com_feat.shape[0], -1)
                relative_scores = self.group.inference(leaf_probs, delta)

                pred_score = relative_scores + l2
                if self.usingDD:
                    pred_score = pred_score * batch_data["difficulty"]

                avg_pred_score += pred_score / len(label_2)

            outputs = {
                "preds": avg_pred_score,
                "labels": batch_data["score"],
            }

        return outputs
