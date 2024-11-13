#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/10 11:48:50


import os
import numpy as np
from scipy import stats
import csv


class Evaluator(object):
    """
    Evaluator for evaluation.
    """

    def __init__(self, log_dir, phase):
        """
        Init evaluator.
        """
        self.rl2e_list = []
        self.srcc_list = []
        self.l2_list = []
        self.best_srcc = -1
        self.best_rl2e = -1
        self.best_l2 = -1
        self.best_srcc_epoch = -1
        self.best_rl2e_epoch = -1
        self.best_l2_epoch = -1
        # temp variables for each epoch
        self.preds = []
        self.labels = []
        # metric log file
        self.metrics_file = os.path.join(log_dir, f"{phase}-metrics.csv")
        self.preds_file = os.path.join(log_dir, f"{phase}-best-preds.csv")
        # number of samples
        self.num_samples = 0

    def add_data(self, outputs):
        """
        Add data.
        """
        preds, labels = outputs["preds"], outputs["labels"]
        # detach and move data to cpu
        preds = preds.detach().cpu().numpy()
        preds = preds.reshape(-1)
        labels = labels.cpu().numpy()
        labels = labels.reshape(-1)

        self.preds.extend(preds)
        self.labels.extend(labels)

        self.num_samples = len(self.labels)

    def calculate_metrics(self):
        """
        Evaluate.
        """
        pred_scores = self.preds
        true_scores = self.labels

        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = (
            np.power(
                (pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2
            ).sum()
            / true_scores.shape[0]
            * 100
        )

        self.rl2e = RL2
        self.srcc = rho
        self.l2 = L2

        self.rl2e_list.append(RL2)
        self.srcc_list.append(rho)
        self.l2_list.append(L2)

        self.best_srcc = max(self.srcc_list)
        self.best_rl2e = min(self.rl2e_list)
        self.best_l2 = min(self.l2_list)

        self.best_srcc_epoch = self.srcc_list.index(self.best_srcc)
        self.best_rl2e_epoch = self.rl2e_list.index(self.best_rl2e)
        self.best_l2_epoch = self.l2_list.index(self.best_l2)

        # write metrics log to a csv file
        self.write_metrics_file(self.metrics_file)
        if self.best_srcc_epoch == len(self.srcc_list) - 1:
            self.write_preds_file()

        # temp variables for each epoch
        self.preds = []
        self.labels = []

    def write_preds_file(self, preds_file=None):
        """
        Write and add preds to a csv file.
        """
        if preds_file is None:
            preds_file = self.preds_file

        with open(preds_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["preds", "labels"])

        with open(preds_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(zip(self.preds, self.labels))

    def write_metrics_file(self, metrics_file):
        """
        Write and add metrics log newline to a csv file.
        epoch srcc best_srcc best_srcc_epoch l2 best_l2 best_l2_epoch rl2e best_rl2e best_rl2e_epoch
        """
        if not os.path.exists(metrics_file):
            with open(metrics_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "epoch",
                        "srcc",
                        "best_srcc",
                        "best_srcc_epoch",
                        "l2",
                        "best_l2",
                        "best_l2_epoch",
                        "rl2e",
                        "best_rl2e",
                        "best_rl2e_epoch",
                    ]
                )

        with open(metrics_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    len(self.srcc_list),
                    self.srcc,
                    self.best_srcc,
                    self.best_srcc_epoch + 1,
                    self.l2,
                    self.best_l2,
                    self.best_l2_epoch + 1,
                    self.rl2e,
                    self.best_rl2e,
                    self.best_rl2e_epoch + 1,
                ]
            )
