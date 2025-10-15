import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # 上一级
# 如果 util 在上上级：再加一层 ".."
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import argparse
import string 
from box import Box
import yaml
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from util import variance_scaling_init, enhance_distribution


class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, embed_dim,  weights=None, alpha=1.0, device="cpu"):
        super().__init__()
        self.n_clusters = n_clusters
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.register_parameter("clusters", None)
        if weights is not None:
            self.set_weights(weights, device=device)
        else:   
            self.build()

    @torch.no_grad()
    def set_weights(self, weights, device=None):
        w = torch.as_tensor(weights, device=device)
        if w.ndim != 2 or w.size(0) != self.n_clusters:
            raise ValueError(f"weights shape must be (n_clusters, n_features); got {tuple(w.shape)}")
        self.clusters = nn.Parameter(w.clone(), requires_grad=True)

    def build(self):
        if self.clusters is None:
            K, D = self.n_clusters, self.embed_dim
            param = torch.empty((K, D))
            nn.init.xavier_uniform_(param)  # glorot_uniform
            self.clusters = nn.Parameter(param, requires_grad=True)

    def forward(self, inputs):
        #q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, dim=1) - self.clusters), dim=2) / self.alpha))
        q = 1.0 / (1.0 + torch.sum((inputs.unsqueeze(1) - self.clusters.unsqueeze(0)) ** 2, dim=2) / self.alpha)
        q **= (self.alpha + 1.0) / 2.0
        q = q / q.sum(dim=1, keepdim=True)
        # q = enhance_distribution(q, 2)
        return q

# class ClusteringLayer(nn.Module):

#     def __init__(self, n_clusters, weights=None, alpha=1.0, device="cpu"):
#         super().__init__()
#         self.n_clusters = n_clusters
#         self.alpha = alpha
#         self.register_parameter("clusters", None)
#         if weights is not None:
#             self.set_weights(weights, device=device)

#     @torch.no_grad()
#     def set_weights(self, weights, device=None):
#         w = torch.as_tensor(weights, device=device)
#         if w.ndim != 2 or w.size(0) != self.n_clusters:
#             raise ValueError(f"weights shape must be (n_clusters, n_features); got {tuple(w.shape)}")
#         self.clusters = nn.Parameter(w.clone(), requires_grad=True)

#     def build(self, x: torch.Tensor):
#         """
#         若未提供初始中心，则在第一次 forward 根据特征维度初始化
#         """
#         if self.clusters is None:
#             K, D = self.n_clusters, x.size(1)
#             param = torch.empty((K, D), dtype=x.dtype, device=x.device)
#             nn.init.xavier_uniform_(param)  # glorot_uniform
#             self.clusters = nn.Parameter(param, requires_grad=True)

#     def forward(self, inputs):
#         self.build(inputs)
#         q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, dim=1) - self.clusters), dim=2) / self.alpha))
#         q **= (self.alpha + 1.0) / 2.0
#         q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, dim=1))
#         return q