import torch
import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import distinctipy
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm
import seaborn as sns

def variance_scaling_init(tensor, scale=1./3., mode='fan_in', distribution='uniform'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        n = fan_in
    elif mode == 'fan_out':
        n = fan_out
    elif mode == 'fan_avg':
        n = (fan_in + fan_out) / 2.
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if distribution == 'uniform':
        limit = math.sqrt(scale / n)
        nn.init.uniform_(tensor, -limit, limit)
    elif distribution == 'normal':
        std = math.sqrt(scale / n)
        nn.init.normal_(tensor, mean=0., std=std)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")

def student_distribution(inputs, centers):
        # with torch.no_grad():
            alpha = 1
            q = 1.0 / (1.0 + torch.sum((inputs.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2) / alpha)
            q **= (alpha + 1.0) / 2.0
            q = q / q.sum(dim=1, keepdim=True)
            return q

# def enhance_distribution(p):
#     weight = p ** 2 / p.sum(0)
#     return (weight.T / weight.sum(1)).T


def enhance_distribution(p, alpha=2.0, eps=1e-12):
    p = p.clamp_min(eps)
    w = p ** alpha
    return w / w.sum(dim=1, keepdim=True)


def plot_tsne(features, cluster_labels, save_dir, seed):
    cluster_labels = LabelEncoder().fit_transform(cluster_labels)
    n_classes = len(set(cluster_labels))
    colors = sns.color_palette("hls", n_classes)  # 可以生成任意数量
    # colors = cm.rainbow(np.linspace(0, 1, n_classes))
    # 1. 降维
    tsne = TSNE(n_components=2, random_state= seed)
    features_2d = tsne.fit_transform(features)  # shape: (N, 2)
    # 2. 可视化
    
    plt.figure(figsize=(8, 6))
    #scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=[colors[l] for l in cluster_labels], s=2)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='tab10', s=20)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.savefig(save_dir, dpi=600, bbox_inches='tight', pad_inches=0)
    print(f'visual figure has saved to {save_dir}')
    plt.grid(False)
    plt.tight_layout()
    plt.show()