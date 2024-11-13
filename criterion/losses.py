#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/10 19:33:13
import torch
import torch.nn as nn


class kld(nn.Module):
    def __init__(self, args):
        super(kld, self).__init__()

        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, outputs):
        preds = outputs["loss_preds"]
        targets = outputs["loss_labels"]
        # Preds are alreadly in probability form and ensure preds are in log form
        preds_log_prob = torch.log(preds).reshape(-1, preds.size(-1))
        # The targets is already in probability form
        targets_prob = targets.reshape(-1, preds.size(-1))
        loss = self.criterion(preds_log_prob, targets_prob)
        return loss


class nll_mse(nn.Module):
    def __init__(self, args):
        super(nll_mse, self).__init__()

        depth = args.head_args["depth"]
        self.num_leaf = 2 ** (depth - 1)

        self.nll = nn.NLLLoss()
        self.mse = nn.MSELoss()

    def forward(self, outputs):
        leaf_probs_1 = outputs["leaf_probs_1"]
        leaf_probs_2 = outputs["leaf_probs_2"]
        glabel_1 = outputs["glabel_1"]
        glabel_2 = outputs["glabel_2"]
        rlabel_1 = outputs["rlabel_1"]
        rlabel_2 = outputs["rlabel_2"]
        delta_1 = outputs["delta_1"]
        delta_2 = outputs["delta_2"]

        loss = 0
        loss += self.nll(leaf_probs_1, glabel_1.argmax(1))
        loss += self.nll(leaf_probs_2, glabel_2.argmax(1))
        for i in range(self.num_leaf):
            mask = rlabel_1[:, i] >= 0
            if mask.sum() != 0:
                loss += self.mse(
                    delta_1[:, i][mask].reshape(-1, 1).float(),
                    rlabel_1[:, i][mask].reshape(-1, 1).float(),
                )
            mask = rlabel_2[:, i] >= 0
            if mask.sum() != 0:
                loss += self.mse(
                    delta_2[:, i][mask].reshape(-1, 1).float(),
                    rlabel_2[:, i][mask].reshape(-1, 1).float(),
                )

        return loss


class mse(nn.Module):
    def __init__(self, args):
        super(mse, self).__init__()

        self.criterion = nn.MSELoss()

    def forward(self, outputs):
        preds = outputs["loss_preds"]
        targets = outputs["loss_labels"]

        return self.criterion(preds, targets)


class Criterion(nn.Module):
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.args = args
        if args.criterion == "mse":
            self.criterion = nn.MSELoss()
        elif args.criterion == "kld":
            self.criterion = kld()
        elif args.criterion == "nll_mse":
            self.criterion = nll_mse(depth=args.head_args["depth"])
        else:
            raise NotImplementedError

    def forward(self, outputs):

        if self.args.criterion == "mse":
            preds = outputs["loss_preds"]
            targets = outputs["loss_labels"]
            return self.criterion(preds, targets)
        else:
            return self.criterion(outputs)
