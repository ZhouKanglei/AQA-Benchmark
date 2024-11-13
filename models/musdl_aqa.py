#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/12 17:15:21

import torch

from utils.misc import normal_pdf
from .aqa import AQA


class MUSDL_AQA(AQA):
    def __init__(self, args):
        super(MUSDL_AQA, self).__init__(args)
        self.label_max = (
            self.args.label_max if hasattr(self.args, "label_max") else None
        )
        self.label_min = (
            self.args.label_min if hasattr(self.args, "label_min") else None
        )
        self.judge_max = (
            self.args.judge_max if hasattr(self.args, "judge_max") else None
        )
        self.judge_min = (
            self.args.judge_min if hasattr(self.args, "judge_min") else None
        )

        self.model_type = "MUSDL" if args.head_args["num_judges"] > 1 else "USDL"
        output_dims = {"USDL": 101, "MUSDL": 21}
        self.output_dim = output_dims[self.model_type]
        self.std = args.std

        self.max = self.judge_max if self.model_type == "MUSDL" else self.label_max

    def label2soft(self, data):
        # Define the range for the soft labels
        label_range = torch.arange(self.output_dim)

        if self.model_type == "USDL":
            labels = data["score"]
            label_range = label_range.unsqueeze(0).to(labels.device)
        else:
            labels = data["judge_scores"]
            label_range = label_range.unsqueeze(0).unsqueeze(0).to(labels.device)

        # Compute soft labels for each label in the batch
        labels_ = labels.unsqueeze(-1) * (self.output_dim - 1) / self.max
        soft_labels = normal_pdf(label_range, labels_, self.std)

        soft_labels = soft_labels / soft_labels.sum(dim=-1, keepdim=True)
        soft_labels = soft_labels.to(torch.float32)

        return soft_labels

    def soft2label(self, probs, data):

        if self.model_type == "USDL":
            pred = probs.argmax(dim=-1) * (self.max / (self.output_dim - 1))
        else:
            # calculate expectation & denormalize & sort
            judge_scores_pred = probs.argmax(dim=-1) * self.max / (self.output_dim - 1)
            judge_scores_pred = judge_scores_pred.sort()[0]  # N, 7

            # keep the median 3 scores to get final score according to the rule of diving
            pred = torch.sum(judge_scores_pred[:, 2:5], dim=1) * data["difficulty"]
        return pred

    def forward(self, batch_data):
        video, label = batch_data["video"], batch_data["score"]
        soft_label = self.label2soft(batch_data)

        features = self.get_clip_features(video)
        h = self.neck(features)
        pred_soft_label = self.head(h)
        pred = self.soft2label(pred_soft_label, batch_data)

        outputs = {
            "preds": pred,
            "labels": label,
            "loss_preds": pred_soft_label,
            "loss_labels": soft_label,
        }

        return outputs
