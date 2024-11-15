#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/13 22:43:30

import torch
import torch.nn as nn

from .triplet_loss import HardTripletLoss


class DRLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0, reg_lambda=0.0):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
        self,
        feat,
        target,
        h_norm2=None,
        m_norm2=None,
        avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight


def choose_align(type_id, n, device, b):
    if type_id == 0:
        return torch.arange(n, device=device).repeat(b)  # 0 - n
    elif type_id == 1:
        return (torch.arange(-n // 2, n // 2, 1, device=device) / (n // 2)).repeat(b)
    else:
        n = n * b
        return torch.arange(-n // 2, n // 2, 1, device=device) / (n // 2)  # -1, 1


class cofinal_loss(nn.Module):
    def __init__(self, args):
        super(cofinal_loss, self).__init__()
        alpha = args.criterion_args["alpha"]
        margin = args.criterion_args["margin"]
        loss_align = args.criterion_args["loss_align"]

        self.dr_loss = nn.L1Loss()  # DRLoss()
        self.loss_align = loss_align
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha
        self.beta = 10

    def forward(self, outputs):
        gt_label = outputs["labels"]
        pred = outputs["loss_preds"]
        label = outputs["loss_labels"]
        feat = outputs["q1"]

        if feat is not None:
            t_loss = 0
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.view(-1, c)  # (bn, c)
            la = choose_align(self.loss_align, n, device, b)
            t_loss = self.triplet_loss(flat_feat, la)  # t_loss = 0

        else:
            self.alpha = 0
            t_loss = 0

        mse_loss = 0
        mse_loss += self.mse_loss(pred["score"][0], gt_label) / 3
        mse_loss += self.mse_loss(pred["score"][1], gt_label) / 3
        mse_loss += self.mse_loss(pred["score"][2], gt_label) / 3

        dr_loss = 0
        dr_loss += self.dr_loss(pred["etf_feat"][0], label[0])
        dr_loss += self.dr_loss(pred["etf_feat"][1], label[1])

        return dr_loss + self.alpha * t_loss + mse_loss