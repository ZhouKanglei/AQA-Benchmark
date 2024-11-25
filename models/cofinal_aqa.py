#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/13 22:39:21

from .aqa import AQA


class COFINAL_AQA(AQA):

    def denormalize(self, outputs):
        # only denormalize the preds and labels for evaluation not for the loss calculation
        outputs["preds"] = outputs["preds"] * self.score_range
        outputs["labels"] = outputs["labels"] * self.score_range
        # outputs["loss_preds"] = [o * self.score_range for o in outputs["loss_preds"]]
        # outputs["loss_labels"] = [o * self.score_range for o in outputs["loss_labels"]]

    def forward(self, batch_data):
        if self.norm_score:
            self.normalize(batch_data)
        video, label = batch_data["video"], batch_data["score"]

        features = self.get_clip_features(video)
        h = self.neck(features)
        out = self.head(h)
        pred = out["output"]["score"][0]

        if self.usingDD:
            loss_etf_labels = self.head.get_proj_class(label / batch_data["difficulty"])
            pred = pred * batch_data["difficulty"]
            out["output"]["score"] = [
                p * batch_data["difficulty"] for p in out["output"]["score"]
            ]
        else:
            loss_etf_labels = self.head.get_proj_class(label)

        outputs = {
            "preds": pred,
            "labels": label,
            "loss_preds": out["output"]["score"],
            "loss_labels": self.head.gen_label_score(label),
            "loss_etf_preds": out["output"]["etf_feat"],
            "loss_etf_labels": loss_etf_labels,
            "q1": out["embed"],
        }

        if self.norm_score:
            self.denormalize(outputs)

        return outputs
