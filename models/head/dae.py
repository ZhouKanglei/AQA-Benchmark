#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/07 21:56:59

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAE(nn.Module):
    def __init__(self, input_dim=1024):
        super(DAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fch = nn.Linear(256, 128)
        self.fc2_mean = nn.Linear(128, 1)
        self.fc2_logvar = nn.Linear(128, 1)

    def encode(self, x):
        h0 = F.relu(self.fc1(x))
        h1 = F.relu(self.fch(h0))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * esp if self.training else mu
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return z.view(-1)
