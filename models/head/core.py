#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/12 17:03:31

import torch.nn as nn
import torch.nn.functional as F


class CORE(nn.Module):
    def __init__(self, in_channel=2049, hidden_channel=256, depth=5):
        super(CORE, self).__init__()
        self.depth = depth
        self.num_leaf = 2 ** (depth - 1)

        self.first_layer = nn.Sequential(
            nn.Linear(in_channel, hidden_channel), nn.ReLU(inplace=True)
        )

        self.feature_layers = nn.ModuleList(
            [self.get_tree_layer(2**d, hidden_channel) for d in range(self.depth - 1)]
        )
        self.clf_layers = nn.ModuleList(
            [self.get_clf_layer(2**d, hidden_channel) for d in range(self.depth - 1)]
        )
        self.reg_layer = nn.Conv1d(
            self.num_leaf * hidden_channel, self.num_leaf, 1, groups=self.num_leaf
        )

    @staticmethod
    def get_tree_layer(num_node_in, hidden_channel=256):
        return nn.Sequential(
            nn.Conv1d(
                num_node_in * hidden_channel,
                num_node_in * 2 * hidden_channel,
                1,
                groups=num_node_in,
            ),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def get_clf_layer(num_node_in, hidden_channel=256):
        return nn.Conv1d(
            num_node_in * hidden_channel, num_node_in * 2, 1, groups=num_node_in
        )

    def forward(self, input_feature):

        out_prob = []
        x = self.first_layer(input_feature)
        bs = x.size(0)
        x = x.unsqueeze(-1)
        for i in range(self.depth - 1):
            prob = self.clf_layers[i](x).squeeze(-1)
            x = self.feature_layers[i](x)
            # print(prob.shape,x.shape)d
            if len(out_prob) > 0:
                prob = F.log_softmax(prob.view(bs, -1, 2), dim=-1)
                pre_prob = out_prob[-1].view(bs, -1, 1).expand(bs, -1, 2).contiguous()
                prob = pre_prob + prob
                out_prob.append(prob)
            else:
                out_prob.append(
                    F.log_softmax(prob.view(bs, -1, 2), dim=-1)
                )  # 2 branch only
        delta = self.reg_layer(x).squeeze(-1)

        return out_prob, delta
