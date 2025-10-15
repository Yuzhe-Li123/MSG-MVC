import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from chebyKANLayer import ChebyKANLinear

class Encoder(nn.Module):
    def __init__ (self, dims, act = 'relu'):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)            
        ])
        # for i in range(len(self.layers)):
            # variance_scaling_init(self.layers[i].weight, scale=1./3., mode='fan_in', distribution='uniform')
            # nn.init.zeros_(self.layers[i].bias)
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif act == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {act}")
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 最后一层不激活
                x = self.activation(x)
        return x
    

class Decoder(nn.Module):
    def __init__ (self, dims, act = 'relu'):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)            
        ])
        # for i in range(len(self.layers)):
            # variance_scaling_init(self.layers[i].weight, scale=1./3., mode='fan_in', distribution='uniform')
            # nn.init.zeros_(self.layers[i].bias)
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif act == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {act}")
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 最后一层不激活
                x = self.activation(x)
        return x

if __name__ == "__main__":
    encoder = Encoder([1,2,3,4])
