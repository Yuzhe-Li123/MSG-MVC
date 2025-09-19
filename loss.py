import numpy as np
from time import time
import Nmetrics
import matplotlib.pyplot as plt
import random
from dataset import MultiViewDataset
import yaml
from box import Box
import string
import cupy as cp
from cuml.cluster import KMeans as cuKMeans
import torch
from torch.utils.data import DataLoader
from models.MSGMVC import MSGMVC
import cupy as cp
import torch.nn as nn
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import Nmetrics
from util import enhance_distribution, student_distribution
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import math

def contrastive_loss_row(y_true: torch.Tensor,
                           y_pred: torch.Tensor,
                           tau: float = 1,
                           eps: float = 1e-12):
    """
    """
    # 防止概率为0

    P = torch.clamp(y_true, min=eps)
    Q = torch.clamp(y_pred, min=eps)
    P = P / P.sum(dim=1, keepdim=True)
    Q = Q / Q.sum(dim=1, keepdim=True)
    Q_log = torch.log(Q + eps)
    P_log = torch.log(P + eps)
    N = P.size(0)
    loss1 = -(P * Q).sum(dim=1).mean() / tau
    loss2 = -(Q * P).sum(dim=1).mean() / tau

    # symmetric
    return (loss1 + loss2) / 2



# def contrastive_loss_row(y_true: torch.Tensor,
#                            y_pred: torch.Tensor,
#                            tau: float = 1,
#                            eps: float = 1e-12):
#     """
#     """
#     # 防止概率为0

#     P = torch.clamp(y_true, min=eps)
#     Q = torch.clamp(y_pred, min=eps)
#     P = P / P.sum(dim=1, keepdim=True)
#     Q = Q / Q.sum(dim=1, keepdim=True)
#     Q_log = torch.log(Q + eps)
#     P_log = torch.log(P + eps)
#     N = P.size(0)
#     targets = torch.arange(N, device=P.device)

#     # view1 -> view2
#     logits1 = (P @ Q_log.t())/tau
#     # loss = F.nll_loss(logits, targets, reduction="mean") 
#     loss1 = F.cross_entropy(logits1, targets, reduction="mean")
    
#     # view2 -> view1YTF10
#     logits2 = (Q @ P_log.t()) / tau
#     loss2 = F.cross_entropy(logits2, targets, reduction="mean")

#     # symmetric
#     return (loss1 + loss2) / 2


def contrastive_loss_column(y_true: torch.Tensor,
                           y_pred: torch.Tensor,
                           tau: float = 1,
                           eps: float = 1e-12):
    """
    """
    # 防止概率为0
    P = y_true.t()
    Q = y_pred.t()
    P = torch.clamp(P, min=eps)
    Q = torch.clamp(Q, min=eps)
    P = P / P.sum(dim=1, keepdim=True)
    Q = Q / Q.sum(dim=1, keepdim=True)
    Q_log = torch.log(Q + eps)
    P_log = torch.log(P + eps)
    N = P.size(0)
    targets = torch.arange(N, device=P.device)

    # view1 -> view2
    logits1 = (P @ Q_log.t())/tau
    # loss = F.nll_loss(logits, targets, reduction="mean") 
    loss1 = F.cross_entropy(logits1, targets, reduction="mean")
    
    # view2 -> view1YTF10
    logits2 = (Q @ P_log.t()) / tau
    loss2 = F.cross_entropy(logits2, targets, reduction="mean")

    # symmetric
    return (loss1 + loss2) / 2


def mimvc_loss(
    x,
    z,
    features,
    reconstructed_x, 
    reconstructed_z, 
    cluster_unique_assign, 
    cluster_sp_assign,
    args
):
    eps = 1e-15
    # cluster_unique_assign_log = (cluster_unique_assign + eps).log()
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    cluster_sp_assign_log = [torch.log(cluster_sp_assign[v]) for v in range(len(x))]
    cluster_unique_assign_log = torch.log(cluster_unique_assign)
    losses_sic = [mse_loss(x[v], reconstructed_x[v]) for v in range(len(x))]
    losses_rci = [
        0.5 * mse_loss(features[v].detach(), reconstructed_z[v]) + 0.5 *  mse_loss(reconstructed_z[v].detach(), features[v]) for v in range(len(x))
    ]
    # losses_rci = [
    #     0.5 * mse_loss(features[v].detach(), reconstructed_z) + 0.5 *  mse_loss(reconstructed_z.detach(), features[v]) for v in range(len(x))
    # ]
    losses_cca_row = [contrastive_loss_row(cluster_unique_assign, cluster_sp_assign[v]) for v in range(len(x))]
    losses_cca_column = [contrastive_loss_column(cluster_unique_assign, cluster_sp_assign[v]) for v in range(len(x))]
    # losses_kl = [(kl_loss(cluster_sp_assign_log[v],  cluster_unique_assign) + kl_loss(cluster_unique_assign_log, cluster_unique_assign[v]))/2 for v in range(len(x))]
    w = [args.ae_weight, args.dg_weight, args.contrastive_weight_row, args.contrastive_weight_column]
    loss = []
    for v in range(len(x)):
        loss_v = w[0] * losses_sic[v] + w[1] * losses_rci[v] + w[2] * losses_cca_row[v] + w[3] * losses_cca_column[v] 
        loss.append(loss_v)
    return loss

