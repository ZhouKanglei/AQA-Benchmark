#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/13 22:39:21

from .aqa import AQA


class COFINAL_AQA(AQA):

    def forward(self, batch_data):
        video, label = batch_data["video"], batch_data["score"]

        features = self.get_clip_features(video)
        h = self.neck(features)
        out = self.head(h)
        pred = out["output"]["score"][0]

        if self.usingDD:
            loss_labels = self.head.get_proj_class(label / batch_data["difficulty"])
            pred = pred * batch_data["difficulty"]
            out["output"]["score"] = [p * batch_data["difficulty"] for p in out["output"]["score"]]
        else:
            loss_labels = self.head.get_proj_class(label)

        outputs = {
            "preds": pred,
            "labels": label,
            "loss_preds": out["output"],
            "loss_labels": loss_labels,
            "q1": out["embed"],
        }

        return outputs
