#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/25 22:13:37

import torch
import torch.nn as nn

from .dtw_loss import SoftDTW


def simclr_loss(output_fast, output_slow, Temperature=0.1, normalize=True):
    out = torch.cat((output_fast, output_slow), dim=0)
    sim_mat = torch.mm(out, torch.transpose(out, 0, 1))
    if normalize:
        sim_mat_denom = torch.mm(
            torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(1).t()
        )
        sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
    sim_mat = torch.exp(sim_mat / Temperature)
    if normalize:
        sim_mat_denom = torch.norm(output_fast, dim=1) * torch.norm(output_slow, dim=1)
        sim_match = torch.exp(
            torch.sum(output_fast * output_slow, dim=-1) / sim_mat_denom / Temperature
        )
    else:
        sim_match = torch.exp(
            torch.sum(output_fast * output_slow, dim=-1) / Temperature
        )
    sim_match = torch.cat((sim_match, sim_match), dim=0)
    norm_sum = torch.exp(torch.ones(out.size(0)) / Temperature)
    norm_sum = norm_sum.to(sim_match.device)
    loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
    return loss


class t2cr_loss(nn.Module):
    def __init__(self, args):
        super(t2cr_loss, self).__init__()
        self.args = args

        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        self.mse = nn.MSELoss()

    def forward(self, outs):
        logits1 = outs["logits1"]
        logits2 = outs["logits2"]
        logits3 = outs["logits3"]
        logits4 = outs["logits4"]

        label_1 = outs["label_1"]
        label_2 = outs["label_2"]

        pred1 = outs["pred1"]
        pred2 = outs["pred2"]

        pred_self1 = outs["pred_self1"]
        pred_self2 = outs["pred_self2"]

        preds1 = outs["preds1"]
        preds2 = outs["preds2"]
        preds3 = outs["preds3"]
        preds4 = outs["preds4"]

        score1 = outs["score1"]
        score2 = outs["score2"]
        score3 = outs["score3"]
        score4 = outs["score4"]

        zero_lable = torch.zeros_like(pred_self1).to(pred1.device)
        loss1 = self.mse(logits1, logits3) + self.mse(logits2, logits4)
        loss2 = simclr_loss(logits1.mean(1), logits3.mean(1)) + simclr_loss(
            logits2.mean(1), logits4.mean(1)
        )
        loss3 = self.mse(pred1, label_1 - label_2) + self.mse(pred2, label_2 - label_1)
        loss4 = (
            self.mse(preds1, label_1)
            + self.mse(preds2, label_2)
            + self.mse(preds3, label_1)
            + self.mse(preds4, label_2)
        )
        loss5 = (
            self.mse(score1, label_1)
            + self.mse(score2, label_2)
            + self.mse(score3, label_1)
            + self.mse(score4, label_2)
        )
        loss6 = 0
        loss7 = 10.0 * (
            self.mse(pred_self1, zero_lable) + self.mse(pred_self2, zero_lable)
        )
        # loss8 = self.sdtw(logits1, logits2).mean() + self.sdtw(logits3, logits4).mean()
        loss8 = 0
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
        return loss
