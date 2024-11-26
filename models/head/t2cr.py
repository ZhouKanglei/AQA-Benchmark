#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/25 20:02:22
import torch
import torch.nn as nn

from ..utils.target_aware_attention import TAA
from ..utils.vit_decoder import decoder_fuser


class SELayer_1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_1d, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MLP_score(nn.Module):
    def __init__(self, in_channel=64, out_channel=1):
        super(MLP_score, self).__init__()
        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Linear(in_channel, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, out_channel)
        self.selayer = SELayer_1d(in_channel)

    def forward(self, x):
        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        x = self.selayer.forward(x.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.layer3(x)

        return output


class T2CR(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        hidden_dim=64,
        num_heads=8,
        num_layers=3,
        thin_dim=8,
        full_dim=10,
    ):
        super(T2CR, self).__init__()

        self.taa = TAA(conv_input=in_channels, thin_dim=thin_dim, full_dim=full_dim)

        self.decoder = decoder_fuser(
            dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            full_dim=full_dim,
        )

        self.regressor = MLP_score(in_channel=hidden_dim, out_channel=1)

    def forward(self, target, exemplar):

        logits1, logits2, score1, score2, pred_class1, pred_class2 = self.taa(target)
        logits3, logits4, score3, score4, pred_class3, pred_class4 = self.taa(exemplar)
        logits_1 = torch.cat([logits1, logits3], dim=1)
        logits_2 = torch.cat([logits2, logits4], dim=1)

        decoder_12 = self.decoder(logits_1, logits_2)
        decoder_21 = self.decoder(logits_2, logits_1)
        decoder_1 = self.decoder(logits1, logits3)
        decoder_2 = self.decoder(logits2, logits4)
        decoder_all = torch.cat((decoder_12, decoder_21), dim=0)
        decoder_self = torch.cat((decoder_1, decoder_2), dim=0)

        total = torch.cat([logits1, logits2, logits3, logits4], dim=0)
        pred = self.regressor(decoder_all)
        pred_self = self.regressor(decoder_self)
        preds = self.regressor(total)

        pred = pred.mean(1)
        preds = preds.mean(1)
        pred_self = pred_self.mean(1)

        preds1, preds2, preds3, preds4 = torch.chunk(preds, 4, dim=0)
        pred1, pred2 = torch.chunk(pred, 2, dim=0)
        pred_self1, pred_self2 = torch.chunk(pred_self, 2, dim=0)

        out = {
            "preds1": preds1,
            "preds2": preds2,
            "preds3": preds3,
            "preds4": preds4,
            "pred1": pred1,
            "pred2": pred2,
            "pred_self1": pred_self1,
            "pred_self2": pred_self2,
            "logits1": logits1,
            "logits2": logits2,
            "logits3": logits3,
            "logits4": logits4,
            "score1": score1,
            "score2": score2,
            "score3": score3,
            "score4": score4,
        }
        
        for k, v in out.items():
            if v.shape[1] == 1:
                out[k] = v.squeeze(1)

        return out
