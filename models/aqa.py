#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/09 17:15:39
from pydoc import locate
import torch
import torch.nn as nn


from utils.misc import print_log, normal_pdf, fix_bn


class AQA(nn.Module):
    def __init__(self, args):
        super(AQA, self).__init__()
        self.args = args
        # backbone
        self.backbone = self.build_model(args.backbone, args.backbone_args)
        self.load_pretrained()
        # neck
        self.neck = self.build_model(args.neck, args.neck_args)
        # head
        self.head = self.build_model(args.head, args.head_args)
        # using difficulty degree for score prediction
        self.usingDD = self.args.usingDD if hasattr(self.args, "usingDD") else False
        # clip index for 16 frames
        self.start_frame_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]
        self.n_clips = len(self.start_frame_idx)

    def build_model(self, model_name, model_args):
        if model_name is not None:
            model = locate("models." + model_name)
            if model_args is not None:
                return model(**model_args)
            else:
                return model()
        else:
            return nn.Identity()

    def fix_bn(self):
        log = "- Fixing the batch normalization layers"
        print_log(log, self.args.logger)
        self.backbone.apply(fix_bn)

    def load_pretrained(self):
        pretrained_path = self.args.backbone_args["pretrained_path"]
        log = "- Loading the pretrained weight from {}".format(pretrained_path)
        print_log(log, self.args.logger)
        self.backbone.load_state_dict(torch.load(pretrained_path))
        if self.args.fix_bn:
            self.fix_bn()

    def get_clip_features(self, x):
        # arrange frames into N clips
        clips = [x[:, :, i : i + 16] for i in self.start_frame_idx]
        video_pack = torch.cat(clips, dim=0)

        # extract features
        feature = self.backbone(video_pack)
        feature = feature.view(self.n_clips, x.shape[0], -1).transpose(0, 1)

        return feature

    def forward(self, batch_data):
        video, label = batch_data["video"], batch_data["score"]

        features = self.get_clip_features(video)
        h = self.neck(features)
        pred = self.head(h)

        if self.usingDD:
            pred = pred * batch_data["difficulty"]

        outputs = {
            "preds": pred,
            "labels": label,
            "loss_preds": pred,
            "loss_labels": label,
        }

        return outputs
