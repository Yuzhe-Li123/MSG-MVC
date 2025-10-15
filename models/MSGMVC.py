import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

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
from util import variance_scaling_init
from autoencoders import Encoder, Decoder
from generator import Generator
from cluser_layer import ClusteringLayer

class MSGMVC(nn.Module):
    def __init__ (
            self, 
            num_samples, 
            n_clusters, 
            view_shape,
            alpha = 1.0,
            encoder_dim = [500, 500, 2000],
            generator_dim = [500, 500],             
            args = None
        ):
        super().__init__()
        self.view_shape = view_shape
        self.view = len(view_shape)
        self.num_samples = num_samples
        self.n_clusters = n_clusters

        self.alpha = alpha
        # 实验中保证每个视角的embed_dims的维度相等
        self.embed_dims = args.embed_dim
        self.z_dim = self.embed_dims[0]
        self.unique_center = torch.zeros((self.n_clusters, self.z_dim))
        #self.unique_center = None
        self.args = args
        decoder_dim = encoder_dim[::-1]
        self.encoders = nn.ModuleList(
            [Encoder([self.view_shape[i]] + encoder_dim + [self.embed_dims[i]]) for i in range(len(self.view_shape))]
        )
        self.decoders = nn.ModuleList(
            [Decoder([self.embed_dims[i]] + decoder_dim + [self.view_shape[i]]) for i in range(len(self.view_shape))]
        )
        self.generator = nn.ModuleList(
            [Generator([self.z_dim] + generator_dim + [self.embed_dims[i]]) for i in range(len(self.view_shape))] 
        )
        # self.generator = Generator([self.z_dim] + generator_dim + [self.z_dim]) 
        self.cluster_layers = nn.ModuleList(
            [ClusteringLayer(self.n_clusters, self.z_dim) for i in range(len(self.view_shape))]
        )
        self.best_indice = {
            'acc': 0.0,
            'nmi': 0.0,
            'ari': 0.0,
            'pur': 0.0,
            'sil': -2.0
        }
    
    def forward(self, x, is_pretrain = False):
        zis = [self.encoders[i](x[i]) for i in range(len(x))]
        reconstructed_x = [self.decoders[i](zis[i]) for i in range(len(x))]
        if is_pretrain == True:
            return reconstructed_x
        cluster_sp = [self.cluster_layers[i](zis[i]) for i in range(len(x))]
        return zis, reconstructed_x, cluster_sp

    def update_best_indice(self, new_indice): 
        for key in self.best_indice.keys():
            if self.best_indice[key] < new_indice[key]:
                self.best_indice = new_indice
                return True
            elif self.best_indice[key] > new_indice[key]:
                return False
        return False
    
    def save_pretrain_model(self):
        torch.save({
            "encoders": self.encoders.state_dict(),
            "decoders": self.decoders.state_dict()
        }, self.args.pretrain_weights)

    def save_model(self):
        torch.save({
            "model": self.state_dict(),
        }, self.args.weights)
    
    def load_pretrain_model(self, device):
        checkpoint = torch.load(self.args.pretrain_weights, map_location=device)
        self.encoders.load_state_dict(checkpoint["encoders"])
        self.decoders.load_state_dict(checkpoint["decoders"])
        self.encoders.to(device)
        self.decoders.to(device)

    # def load_model(self, device):
    #     # torch.serialization.add_safe_globals([MIMVC])
    #     self = torch.load(self.args.weights, map_location=device, weights_only=False)
    #     self.to(device)
    #     self.eval()  # 如果是推理用
    #     return self

    def load_model(self, device):
        checkpoint = torch.load(self.args.weights, map_location=device)
        self.load_state_dict(checkpoint["model"])
        self.to(device)
        self.eval()  # 如果是推理用
        return self
