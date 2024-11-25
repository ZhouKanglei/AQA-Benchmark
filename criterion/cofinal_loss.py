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
        self.using_final_score_loss = (
            args.criterion_args["using_final_score_loss"]
            if hasattr(args.criterion_args, "using_final_score_loss")
            else False
        )

        self.dr_loss = nn.MSELoss()
        self.loss_align = loss_align
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha
        self.beta = 10

    def forward(self, outputs):

        gt_label = outputs["loss_labels"]
        pred = outputs["loss_preds"]

        etf_label = outputs["loss_etf_labels"]
        etf_pred = outputs["loss_etf_preds"]

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
        l1_loss = 0
        for i in range(0 if self.using_final_score_loss else 1, len(pred)):
            mse_loss += self.mse_loss(pred[i], gt_label[i])

        dr_loss = 0
        for i in range(len(etf_pred)):
            dr_loss += self.dr_loss(etf_pred[i], etf_label[i])

        return dr_loss + self.alpha * t_loss + mse_loss + l1_loss
