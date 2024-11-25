#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/13 20:49:50


from .aqa import AQA


class GDLT_AQA(AQA):

    def forward(self, batch_data):
        if self.norm_score:
            self.normalize(batch_data)
        video, label = batch_data["video"], batch_data["score"]

        features = self.get_clip_features(video)
        h = self.neck(features)
        pred, q1 = self.head(h)

        if self.usingDD:
            pred = pred * batch_data["difficulty"]

        outputs = {
            "preds": pred,
            "labels": label,
            "loss_preds": pred,
            "loss_labels": label,
            "q1": q1,
        }

        if self.norm_score:
            self.denormalize(outputs)

        return outputs
