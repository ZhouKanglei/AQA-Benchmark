#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/13 22:24:14


from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from ..utils.transformer import Transformer
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


# code from https://github.com/NeuralCollapseApplications/FSCIL/blob/main/mmfscil/models/ETFHead.py#L17
def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(
        torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.0e-7
    ), "The max irregular value is : {}".format(
        torch.max(
            torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))
        )
    )
    return orth_vec


def etf(in_channels, num_classes, normal=False):
    orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(
        torch.ones(num_classes, num_classes), (1 / num_classes)
    )
    etf_vec = torch.mul(
        torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
        math.sqrt(num_classes / (num_classes - 1)),
    )
    etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
    if normal:
        etf_vec = etf_vec / torch.sum(etf_vec, dim=0, keepdim=True)
    return etf_vec, etf_rect


def choose_activate(n_query, type_id=1):
    activations = {
        0: (torch.linspace(0, 1, n_query), torch.linspace(0, 1, n_query).flip(-1)),
        1: (
            torch.linspace(0, 1, n_query + 1)[1:],
            torch.linspace(0, 1, n_query + 1)[1:].flip(-1),
        ),
        2: (torch.tensor([0.1, 0.2, 0.8, 1]), torch.tensor([1, 1, -1, -1])),
        3: (
            torch.tensor([-1, -0.8, 0.8, 1]),
            torch.tensor([-1, -0.8, 0.8, 1]).flip(-1),
        ),
    }
    return activations.get(
        type_id, (torch.linspace(0, 1, n_query), torch.linspace(0, 1, n_query))
    )


class COFINAL(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        hidden_dim=256,
        n_head=1,
        n_encoder=1,
        n_decoder=2,
        n_query=4,
        dropout=0,
        score_range=1,
        etf_vec_dim=128,
        first_etf_num=10,
        second_etf_num=0,  # not used
        using_neg=True,
        key_len=1,
        activate_type=1,
    ):
        super(COFINAL, self).__init__()
        self.score_range = score_range
        self.using_neg = using_neg
        self.first_etf_num = first_etf_num

        # in_proj part
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
        # query weight activation
        we = choose_activate(n_query, activate_type)
        # positive part
        self.regressor = nn.Linear(hidden_dim, n_query)
        self.weight = we[0] * score_range
        if using_neg:
            # negative part: score_range minus positive part
            self.regressor_neg = nn.Linear(hidden_dim, n_query)
            self.weight_neg = we[1] * score_range

        # etf part
        etf_vec1, _ = etf(etf_vec_dim, first_etf_num)
        self.register_buffer("etf_vec1", etf_vec1)

        self.key_len = key_len

        # etf head
        self.regressori = nn.ModuleList()
        self.regi = nn.ModuleList()
        for _ in range(key_len):
            self.regressori.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_query * hidden_dim, hidden_dim),
                    nn.Linear(hidden_dim, etf_vec_dim),
                )
            )

            self.regi.append(
                torch.nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=4, dropout=dropout
                ),
            )

    def forward(self, x):

        # x (b, t, c)
        result = {
            "pos_s": None,
            "neg_s": None,
            "score": None,
            "etf_feat": [],
        }

        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        q1 = self.transformer.decoder(q, encode_x)  # torch.Size([32, 4, 256])

        self.weight = self.weight.to(x.device)
        s = self.regressor(q1)
        out = self.gen_score(s, self.weight, b)
        result["pos_s"] = out

        if self.using_neg:
            self.weight_neg = self.weight_neg.to(x.device)
            s1 = self.regressor_neg(q1)
            out = self.gen_score(s1, self.weight_neg, b)
            result["neg_s"] = out

        for i in range(self.key_len):
            s_dec, _ = self.regi[i](q1, q1, q1)
            s_dec = self.regressori[i](s_dec)
            norm_d = self.norm_logits(s_dec)
            result["etf_feat"].append(norm_d)

        result["score"] = self.gen_final_score(result)

        return {"output": result, "embed": q1}

    def gen_score(self, s, weight, b):
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n) torch.Size([32, 4, 4])
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return out

    def norm_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def re_proj(self, x):
        return torch.argmax(x, dim=-1)
        # return torch.argmax(x)

    def get_proj_class(self, gt_label1):
        gt_label = (gt_label1 * 100 * self.first_etf_num).long()
        g_2 = gt_label - gt_label // 100 * 100

        target_1 = self.etf_vec1[:, g_2].t()

        return (target_1,)

    def gen_label_score(self, gt_label1):
        gt_label = (gt_label1 * 100 * self.first_etf_num).long()

        g_1 = gt_label // 100 / self.first_etf_num

        if self.using_neg:
            return (gt_label1, g_1, self.score_range - g_1)
        else:
            return (gt_label1, g_1)

    def gen_final_score(self, x):
        # the first part is the positive part plus the negative part
        if self.using_neg:
            score1 = (x["pos_s"] + self.score_range - x["neg_s"]) / 2
        else:
            score1 = x["pos_s"]

        # the second part is the etf part
        first_dict = x["etf_feat"][0] @ self.etf_vec1
        first_c = self.re_proj(first_dict)

        # final score： score1 accounts for the first two decimal places
        score = score1 + first_c / 100 / self.first_etf_num

        if self.using_neg:
            return (score, x["pos_s"], x["neg_s"])
        else:
            return (score, x["pos_s"])
