import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # 屏蔽 TF 后端的 INFO/WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # 不要 oneDNN 提示
os.environ["TF_USE_TRT"] = "0"              # 禁掉 TF-TRT
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")# 屏蔽 Python 层 warning

current_pid = os.getpid()
print(f"当前进程 PID: {current_pid}")

import numpy as np
# from sklearn.manifold import TSNE
from time import time
import Nmetrics
import matplotlib.pyplot as plt

import random
from dataset import MultiViewDataset
import yaml
from box import Box
import string
import cupy as cp
import torch
from torch.utils.data import DataLoader
from models.MSGMVC import MSGMVC
from trainer import Trainer
from loss import mimvc_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 error，隐藏 warning/info
# tf.get_logger().setLevel('ERROR')


def set_seed(seed):
    # Python 内置
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU (单卡/多卡)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 确保 cudnn 可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 一些新的 Torch 版本支持开启确定性算子
    # if hasattr(torch, "use_deterministic_algorithms"):
    #     torch.use_deterministic_algorithms(True)


def substitute_variables(value, variables):
	if isinstance(value, str):  # 只替换字符串
		return string.Template(value).safe_substitute(variables)
	return value  # 其他类型（int, float, bool）不变

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = 'BDGP'
# data = "ALOI"
parser = argparse.ArgumentParser(description='main')
parser.add_argument('-d', '--dataset', default=data,
                    help="which dataset")
parser.add_argument('--config-path', default='./config', type=str)
temp_args = parser.parse_args()
config_dict = {}
config_file = os.path.join(temp_args.config_path, temp_args.dataset + ".yaml")
with open(config_file, 'r') as f:
    config_dict = yaml.safe_load(f)
config_dict = {k: substitute_variables(v, config_dict) for k, v in config_dict.items()}
config_dict = Box(config_dict)

args = argparse.Namespace(**config_dict)

if __name__ == "__main__":
    
    set_seed(args.seed)
    print('+' * 30, ' Parameters ', '+' * 30)
    print(args)
    print('+' * 75)
    multi_view_dataset = MultiViewDataset(args.dataset)
    pre_train_dataloader = DataLoader(multi_view_dataset, batch_size = args.pre_batch_size, shuffle = True, num_workers = 0)
    train_dataloader = DataLoader(multi_view_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
    model = MSGMVC(
        num_samples = len(multi_view_dataset),
        n_clusters = multi_view_dataset.get_num_clusters(),
        view_shape = multi_view_dataset.get_views(), 
        args = args
    ).to(device)
    pre_opt = torch.optim.Adam(
        params = list(model.encoders.parameters()) + list(model.decoders.parameters()), 
        lr = args.pre_lr, 
        weight_decay = 0
    )
    opt = torch.optim.Adam(
        params = model.parameters(),
        lr = args.lr, 
        weight_decay = 0 
    )
    scheduler = CosineAnnealingLR(opt, T_max = args.epochs, eta_min=1e-8)
    trainer = Trainer(
        pre_data_loader = pre_train_dataloader,
        data_loader = train_dataloader,
        model = model,
        pre_opt = pre_opt,
        opt = opt,
        scheduler = scheduler,
        loss_fn = mimvc_loss,
        device = device,
        args = args        
    )
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.testing is True and os.path.exists(args.weights):
        print('test')
        trainer.test()
    else:
        if args.train_ae is False and os.path.exists(args.pretrain_weights):
            model.load_pretrain_model(device)
        else:
            trainer.pre_train()
        print('trian')
        trainer.train()
