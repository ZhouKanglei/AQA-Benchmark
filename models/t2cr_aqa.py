#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/25 21:41:15

import torch
import torch.nn as nn

from .aqa import AQA


class T2CR_AQA(AQA):

    def get_paired_clip_features(self, target, exemplar):

        if self.args.backbone is None:
            feature_1, feature_2 = self.merge_paired_clip_features(target, exemplar)
            return feature_1, feature_2

        # spatiotemporal feature
        total_video = torch.cat((target, exemplar), 0)  # 2N C T H W

        start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]
        start_idx_2 = [0, 12, 24, 36, 48, 60, 72, 84]
        video_pack = torch.cat([total_video[:, :, i : i + 16] for i in start_idx])
        video_pack_2 = torch.cat([total_video[:, :, i : i + 16] for i in start_idx_2])
        video_pack = torch.cat([video_pack, video_pack_2], dim=0)
        total_feature = self.backbone(video_pack)
        # Nt, C, T, H, W = total_feamap.size()
        total_feature = total_feature.reshape(
            len(start_idx) + len(start_idx_2), len(total_video), -1
        ).transpose(0, 1)
        total_feature_full = total_feature[:, :10, :]
        total_feature_thin = total_feature[:, 10:, :]

        feature_1 = self.neck(total_feature_full)
        feature_2 = self.neck(total_feature_thin)

        return feature_1, feature_2

    def merge_paired_clip_features(self, target, exemplar):
        total_feat_full = torch.cat((target, exemplar), 0)
        thin = [total_feat_full[:, i * 2 : i * 2 + 1] for i in range(total_feat_full.shape[1] // 2)]
        
        total_feat_thin = torch.cat(thin, 1)
        
        feature_1 = self.neck(total_feat_full)
        feature_2 = self.neck(total_feat_thin)

        return feature_1, feature_2

    def forward(self, batch_data):

        video_1 = batch_data["video"]  # N, C, T, H, W
        video_2 = batch_data["target_video"]  # N, C, T, H, W

        if self.usingDD:
            label_1 = batch_data["completeness"]
            label_2 = batch_data["target_completeness"]
        else:
            label_1 = batch_data["score"]
            label_2 = batch_data["target_score"]

        if self.training:
            # forward
            feat_1, feat_2 = self.get_paired_clip_features(video_1, video_2)

            out = self.head(feat_1, feat_2)

            score1 = out["pred1"] + label_2
            score2 = out["preds1"]
            pred_score = (score1 + score2) / 2.0

            if self.usingDD:
                pred_score = pred_score * batch_data["difficulty"]

            outputs = {
                "preds": pred_score,
                "labels": batch_data["score"],
                "label_1": label_1,
                "label_2": label_2,
                "pred1": out["pred1"],
                "pred2": out["pred2"],
                "preds1": out["preds1"],
                "preds2": out["preds2"],
                "preds3": out["preds3"],
                "preds4": out["preds4"],
                "logits1": out["logits1"],
                "logits2": out["logits2"],
                "logits3": out["logits3"],
                "logits4": out["logits4"],
                "score1": out["score1"],
                "score2": out["score2"],
                "score3": out["score3"],
                "score4": out["score4"],
                "pred_self1": out["pred_self1"],
                "pred_self2": out["pred_self2"],
            }

        else:
            avg_pred_score = 0
            for v2, l2 in zip(video_2, label_2):
                pred_score = 0
                feat_1, feat_2 = self.get_paired_clip_features(video_1, v2)

                out = self.head(feat_1, feat_2)

                score1 = out["pred1"] + l2
                score2 = out["preds1"]
                pred_score = (score1 + score2) / 2.0

                if self.usingDD:
                    pred_score = pred_score * batch_data["difficulty"]

                avg_pred_score += pred_score / len(label_2)

            outputs = {
                "preds": avg_pred_score,
                "labels": batch_data["score"],
            }

        return outputs
