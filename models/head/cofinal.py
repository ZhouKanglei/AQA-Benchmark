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
        second_etf_num=50,
    ):
        super(COFINAL, self).__init__()
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

        # the original regression head from GDLT
        self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        self.regressor_neg = nn.Linear(hidden_dim, n_query)

        # score interval
        self.weight = torch.linspace(0, score_range, n_query)
        self.weight_neg = torch.linspace(0, score_range, n_query)
        # initialize a learnable penalty weight: the penalty weight is used to penalize the negative part
        self.penalty_weight = nn.Parameter(torch.tensor(0.0))

        # etf head
        self.score_range_1 = score_range
        self.score_range_2 = score_range / first_etf_num
        self.score_range_3 = score_range / first_etf_num / second_etf_num

        # first level etf is used to generate the coarse-grained score
        self.first_etf_num = first_etf_num
        etf_vec1, _ = etf(etf_vec_dim, first_etf_num)
        self.register_buffer("etf_vec1", etf_vec1)

        # second level etf is used to generate the fine-grained score
        self.second_etf_num = second_etf_num
        etf_vec2, _ = etf(etf_vec_dim, second_etf_num)
        self.register_buffer("etf_vec2", etf_vec2)

        # two level etf heads
        self.regressori = nn.ModuleList()
        self.regi = nn.ModuleList()
        for _ in range(2):
            self.regi.append(
                torch.nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=4, dropout=dropout
                ),
            )

            self.regressori.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(4 * hidden_dim, etf_vec_dim),
                )
            )

    def forward(self, x):

        self.weight = self.weight.to(x.device)
        self.weight_neg = self.weight_neg.to(x.device)

        result = {
            "pos_s": None,
            "neg_s": None,
            "etf_feat": [],
        }

        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        q1 = self.transformer.decoder(q, encode_x)  # torch.Size([32, 4, 256])

        # score regressor 1: positive part - negative part
        # positive score regressor: action reponses for the positive part
        s = self.regressor(q1)
        out = self.gen_score(s, self.weight, b)
        result["pos_s"] = out

        # negative score regressor (optional): action reponses for the negative part
        s1 = self.regressor_neg(q1)
        out = self.gen_score(s1, self.weight_neg, b)
        result["neg_s"] = out

        # score classification 2: etf part
        # etf head: action reponses for the whole part
        for i in range(2):
            s_dec, _ = self.regi[i](q1, q1, q1)
            s_dec = self.regressori[i](s_dec)
            norm_d = self.norm_logits(s_dec)
            result["etf_feat"].append(norm_d)

        # final score: (positive part - negative part + etf part) / 2
        # the mixture of the positive part, negative part, and etf part provides a more robust score
        result["score"] = self.gen_final_score(result)

        return {"output": result, "embed": q1}

    def gen_score(self, s, weight, b):
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n, n) -> (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return out

    def norm_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def re_proj(self, x):
        return torch.argmax(x, dim=-1)

    def get_proj_class(self, gt_label):

        g_1 = torch.floor(gt_label / self.score_range_1 * self.first_etf_num)
        g_2_score = gt_label - g_1 / self.first_etf_num * self.score_range_1
        g_2 = torch.floor(g_2_score / self.score_range_2)

        g_1 = g_1.long()
        g_2 = g_2.long()

        target_1 = self.etf_vec1[:, g_1].t()
        target_2 = self.etf_vec2[:, g_2].t()

        target = (target_1, target_2)

        return target

    def gen_final_score(self, x):

        first_dict = x["etf_feat"][0] @ self.etf_vec1
        second_dict = x["etf_feat"][1] @ self.etf_vec2
        first_c = self.re_proj(first_dict)
        second_c = self.re_proj(second_dict)

        first_s = first_c * self.score_range_2
        second_s = second_c * self.score_range_3

        score1 = x["pos_s"] + x["neg_s"] * self.penalty_weight
        score2 = first_s + second_s

        score = (score1 + score2) / 2

        if len(score.shape) > 1:
            score = score.squeeze()
        if len(score1.shape) > 1:
            score1 = score1.squeeze()
        if len(score2.shape) > 1:
            score2 = score2.squeeze()

        return score, score1, score2
