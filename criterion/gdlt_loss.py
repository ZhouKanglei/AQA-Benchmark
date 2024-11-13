#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/13 20:31:23

import torch
from torch import nn

from .triplet_loss import HardTripletLoss


class gdlt_loss(nn.Module):
    def __init__(self, args):
        super(gdlt_loss, self).__init__()

        alpha = args.criterion_args["alpha"]
        margin = args.criterion_args["margin"]

        self.mse_loss = nn.MSELoss()
        # self.mse_loss = nn.L1Loss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha

    def forward(self, outputs):
        pred = outputs["loss_preds"]
        label = outputs["loss_labels"]
        feat = outputs["q1"]
        # feat (b, n, c), x (b, t, c)
        if feat is not None:
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.view(-1, c)  # (bn, c)
            la = torch.arange(n, device=device).repeat(b)

            t_loss = self.triplet_loss(flat_feat, la)
            # t_loss = pair_diversity_loss(feat)
        else:
            self.alpha = 0
            t_loss = 0

        mse_loss = self.mse_loss(pred, label)

        total_loss = mse_loss + self.alpha * t_loss

        return total_loss
