#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/10 17:22:51

import torch.nn as nn


class AVG(nn.Module):
    """
    AVG neck.
    """

    def __init__(self, **kwargs):
        """
        Init AVG neck.
        """
        super(AVG, self).__init__()

    def forward(self, x):
        """
        Forward.
        """
        y = x.mean(dim=1)
        return y
