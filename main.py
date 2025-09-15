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
from models.MIMVC import MIMVC
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


# def _make_data_and_model(args):
#     # prepare dataset
#     x, y = load_data_conv(args.dataset)
#     view = len(x)
#     view_shapes = []
#     Loss = []
#     Loss_weights = []
#     lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
#         initial_learning_rate=args.lr,
#         decay_steps=args.maxiter
#     )

#     # prepare optimizer
#     optimizer = Adam(learning_rate = lr_schedule)
#     # prepare the model
#     n_clusters = len(np.unique(y))
#     # n_clusters = 40   # over clustering
#     print("n_clusters:" + str(n_clusters))
#     # lc = 0.1

#     model = MvDEC(filters=[32, 64, 128, 10],num_samples=y.shape[0],  n_clusters=n_clusters, view_shape=view_shapes, embed_dim = args.embed_dim)

#     model.compile(optimizer=optimizer, loss=Loss, loss_weights=Loss_weights)
#     return x, y, model


# def train(args):
#     # get data and mode
#     #x, y, model = _make_data_and_model(args)
#     # pretraining
#     t0 = time()
#     # 创建该数据集的相关文件夹
#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
#     if args.train_ae is False and os.path.exists(args.pretrain_dir):  # load pretrained weights
#         # model.autoencoder.load_weights(args.pretrain_dir)
#         # model.load_weights(args.pretrain_dir)
#     else:  # train
#         optimizer = Adam(lr=args.pre_lr)
#         print()
#         print('------------------------------pretrain-----------------------------------')
#         print()
#         model.pretrain(x, y, optimizer=optimizer, epochs=args.pretrain_epochs,
#                             batch_size=args.pre_batch_size, save_dir=args.save_dir, verbose=args.verbose)
#         args.pretrain_dir = args.save_dir + '/ae_weights.h5'
#     t1 = time()
#     print("Time for pretraining: %ds" % (t1 - t0))

#     # clustering
#     # DEMVC, IDEC, DEC
#     # y_pred, y_mean_pred = model.fit(arg=args, x=x, y=y, maxiter=args.maxiter,
#     #                                            batch_size=args.batch_size, UpdateCoo=args.UpdateCoo,
#     #                                            save_dir=args.save_dir)
#     # SDMVC
#     model.new_fit(arg=args, x=x, y=y, maxiter=args.maxiter,
#                                     batch_size=args.batch_size, UpdateCoo=args.UpdateCoo,
#                                     save_dir=args.save_dir, args = args)
#     # if y is not None:
#     #     for view in range(len(x)):
#     #         print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
#     #                 (Nmetrics.acc(y, y_pred[view]), Nmetrics.nmi(y, y_pred[view]), Nmetrics.ari(y, y_pred[view])))
#     #     print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
#     #               (Nmetrics.acc(y, y_mean_pred), Nmetrics.nmi(y, y_mean_pred), Nmetrics.ari(y, y_mean_pred)))

#     # t2 = time()
#     # print("Time for pretaining, clustering and total: (%ds, %ds, %ds)" % (t1 - t0, t2 - t1, t2 - t0))
#     # print('='*60)


# def test(args):
#     assert args.testing is True

#     x, y, model = _make_data_and_model(args)
#     model.model.summary()
#     print('Begin testing:', '-' * 60)
#     model.load_weights(args.weights)
#     y_pred, y_mean_pred = model.predict_label(x=x)
#     if y is not None:
#         for view in range(len(x)):
#             print('Final: acc=%.4f, nmi=%.4f, pur=%.4f' %
#                     (Nmetrics.acc(y, y_pred[view]), Nmetrics.nmi(y, y_pred[view]), Nmetrics.pur(y, y_pred[view])))
#         print('Final: acc=%.4f, nmi=%.4f, pur=%.4f' %
#                   (Nmetrics.acc(y, y_mean_pred), Nmetrics.nmi(y, y_mean_pred), Nmetrics.pur(y, y_mean_pred)))
    
#     print('End testing:', '-' * 60)

def substitute_variables(value, variables):
	if isinstance(value, str):  # 只替换字符串
		return string.Template(value).safe_substitute(variables)
	return value  # 其他类型（int, float, bool）不变

import argparse

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    model = MIMVC(
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
    # testing
    # if args.testing:
    #     test(args)
    # else:
    #     train(args)
    #     args.testing = True
    #     print()
    #     print('-------------testing------------------')
    #     test(args)
