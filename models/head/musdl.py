#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/12 17:13:21

import torch
import torch.nn as nn


class MLP_block(nn.Module):

    def __init__(self, output_dim, feature_dim=1024):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        output = self.softmax(self.layer3(x))
        return output


class MUSDL(nn.Module):

    def __init__(self, num_judges=1, **kwargs):
        super(MUSDL, self).__init__()

        if num_judges == 1:
            model_type = "USDL"
        else:
            model_type = "MUSDL"

        self.model_type = model_type

        output_dim_dict = {"USDL": 101, "MUSDL": 21}
        output_dim = output_dim_dict[model_type]

        if model_type == "USDL":
            self.evaluator = MLP_block(output_dim=output_dim)
        else:
            assert num_judges is not None, "num_judges is required in MUSDL"
            self.evaluator = nn.ModuleList(
                [MLP_block(output_dim=output_dim) for _ in range(num_judges)]
            )

    def forward(self, feats_avg):  # data: NCTHW

        if self.model_type == "USDL":
            probs = self.evaluator(feats_avg)  # Nxoutput_dim
        else:
            probs = [
                evaluator(feats_avg) for evaluator in self.evaluator
            ]  # len=num_judges

            probs = torch.stack(probs, dim=1)  # Nxnum_judges x output_dim

        return probs
