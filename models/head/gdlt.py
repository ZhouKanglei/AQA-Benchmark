#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/13 16:52:45

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from ..utils.transformer import Transformer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


class GDLT(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        hidden_dim=256,
        n_head=1,
        n_encoder=1,
        n_decoder=2,
        n_query=4,
        dropout=0.3,
    ):
        super(GDLT, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout,
        )

        self.prototype = nn.Embedding(n_query, hidden_dim)

        self.weight = torch.linspace(0, 1, n_query, requires_grad=False)
        self.regressor = nn.Linear(hidden_dim, n_query)

    def forward(self, x):
        self.weight = self.weight.to(x.device)
        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        q1 = self.transformer.decoder(q, encode_x)

        s = self.regressor(q1)  # (b, n, n)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)

        if len(out.shape) > 1:
            out = out.squeeze()
        return out
