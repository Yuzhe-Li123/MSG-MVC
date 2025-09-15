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
from sklearn.cluster import KMeans as skKMeans
import torch
from torch.utils.data import DataLoader
from models.MIMVC import MIMVC
import cupy as cp
import torch.nn as nn
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import Nmetrics
from util import enhance_distribution, student_distribution, plot_tsne
from sklearn.preprocessing import normalize
from sklearn.metrics import calinski_harabasz_score,silhouette_score, silhouette_samples
import torch.nn.functional as F
import os

def minmax_scale_tensor(x: torch.Tensor, eps=1e-12):
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    return (x - x_min) / (x_max - x_min + eps)


class Trainer():
    def __init__(
        self,
        pre_data_loader,
        data_loader,
        model,
        pre_opt,
        opt,
        scheduler,
        loss_fn,
        device,
        args,
    ):
        self.pre_data_loader = pre_data_loader    
        self.data_loader = data_loader
        self.dataset = data_loader.dataset
        self.model = model
        self.pre_opt = pre_opt
        self.opt = opt
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.args = args
        self.n_clusters = self.dataset.get_num_clusters()
        self.dims = self.dataset.get_views()
        self.views = len(self.dims)
        self.seed = args.seed
    
    def pre_train(self):
        self.model.train()
        pre_train_epoch = self.args.pretrain_epochs   
        crit = nn.MSELoss()
        for i in range(pre_train_epoch):
            loss_sum = [0.0] * self.views
            print(f'epoch: {i + 1}')
            for x, _, _ in self.pre_data_loader:
                x = [xi.to(self.device) for xi in x]
                self.pre_opt.zero_grad()    
                reconstructed_x = self.model(x, is_pretrain = True)
                losses = [crit(x[v], reconstructed_x[v]) for v in range(len(x))]
                loss = sum(losses)
                loss.backward()
                self.pre_opt.step()
                for view in range(self.views):
                    loss_sum[view] += losses[view].item()

            loss_total = sum(loss_sum) / self.views
            loss_sum = [loss_total] + loss_sum
            for view in range(len(loss_sum)):
                loss_sum[view] = loss_sum[view] / len(self.pre_data_loader)
            print(f'loss: {loss_sum}')
            print()
        self.model.save_pretrain_model()

    def extract_features(self):
        '''
        提取整体数据的特征, 用于聚类
        '''
        features_list = [[] for i in range(self.views)]
        data_loader = DataLoader(self.dataset, batch_size = 1024, shuffle = False, num_workers = 0)
        self.model.eval()
        with torch.no_grad():
            for x, _, _ in data_loader:
                for i, xi in enumerate(x):
                    z = self.model.encoders[i](xi.to(self.device))
                    features_list[i].append(z.detach())

        features = [torch.cat(f, dim = 0) for f in features_list]
        self.model.train()
        return features


    def view_sp_cluster(self):
        y_preds = []
        centers = []
        features = self.extract_features()
        if self.args.normalize == 1:
            features = [F.normalize(f, p = 2, dim = 1) for f in features]
        # features = [minmax_scale_tensor(f) for f in features]

        # cuml k-means on gpu 
        # for view in range(self.views):
        #     kmeans = cuKMeans(n_clusters=self.n_clusters, init = 'scalable-k-means++', n_init=100, random_state = self.seed)
        #     t = features[view]
        #     t = t.contiguous().float()
        #     feature_cp = cp.fromDlpack(to_dlpack(t))
        #     y_pred_cp = (kmeans.fit_predict(feature_cp))
        #     y_preds.append(from_dlpack(y_pred_cp.toDlpack()).to(self.device))
        #     cc = kmeans.cluster_centers_
        #     centers.append(from_dlpack(cc.toDlpack()).float().to(self.device))
        

        # sklearn k-means on cpu
        for view in range(self.views):
            kmeans = skKMeans(n_clusters=self.n_clusters, init = 'k-means++', n_init=100, random_state = self.seed)
            t = features[view].contiguous().float()
            x = t.detach().cpu().numpy()
            y_pred_np = kmeans.fit_predict(x)
            y_pred = torch.from_numpy(y_pred_np).to(self.device)
            y_preds.append(y_pred)
            cc = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)  # (K, D)
            centers.append(cc)

        return y_preds, centers, features

    def unique_cluster(self):
        y_pred = []
        features = self.extract_features()
        if self.args.normalize == 1:
            features = [F.normalize(f, p = 2, dim = 1) for f in features]
        # z = sum(w * f for w, f in zip(weight, features))
        z = sum(features) / len(features)
        # z = F.normalize(z, p=2, dim=1)

        # cuml k-means on gpu
        # kmeans = cuKMeans(n_clusters=self.n_clusters, init = 'scalable-k-means++', n_init=100, random_state = self.seed)
        # t = z.contiguous().float()
        # t_cp = cp.fromDlpack(to_dlpack(t))
        # y_pred_cp = (kmeans.fit_predict(t_cp))
        # y_pred.append(from_dlpack(y_pred_cp.toDlpack()).to(self.device))
        # cc = kmeans.cluster_centers_
        # center = from_dlpack(cc.toDlpack()).float().to(self.device)

        # sklearn k-means on cpu
        kmeans = cuKMeans(n_clusters=self.n_clusters, init = 'k-means++', n_init=100, random_state = self.seed)
        t = z.contiguous().float()
        t_np = t.detach().cpu().numpy()
        y_pred_np = (kmeans.fit_predict(t_np))
        y_pred = [torch.from_numpy(y_pred_np).to(self.device)]
        cc = kmeans.cluster_centers_
        center = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)

        return y_pred, center, z

    def init_sp_cluster_centers(self, centers):
        self.model.eval()
        with torch.no_grad():
            for view in range(self.views):
                self.model.cluster_layers[view].clusters.copy_(centers[view])
                self.model.cluster_layers[view].clusters.grad = None
    
    def evaluate_sp_cluster(self, y_pred_k, centers, features):
        '''
        评估各个视角的聚类结果
        '''
        # y_pred_k, centers, features = self.view_sp_cluster()
        y_pred_k = [y_pred_k[i].cpu().numpy() for i in range(self.views)]
        y_pred_list = [[] for i in range(self.views)]
        y_true = self.dataset.y.cpu().numpy()
        # self.init_sp_cluster_centers(centers)
        self.model.eval()
        with torch.no_grad():
            for view in range(self.views):
                for i in range(0, len(self.dataset), 1024):
                    features_batch = features[view][i:i+1024]
                    q = self.model.cluster_layers[view](features_batch)
                    q = enhance_distribution(q)
                    y_pred_list[view].append(torch.argmax(q, dim = 1))
        
        y_pred = [torch.cat(f, dim = 0).cpu().numpy() for f in y_pred_list]
        for view in range(self.views):
            acc = Nmetrics.acc(y_true, y_pred[view])
            nmi = Nmetrics.nmi(y_true, y_pred[view])
            ari = Nmetrics.ari(y_true, y_pred[view])
            pur = Nmetrics.pur(y_true, y_pred[view])
            # sil = silhouette_score(features[view].cpu().numpy(), y_pred[view])
            print(f'View: {view + 1}, acc: {acc:.5f}, nmi: {nmi:.5f}, ari: {ari:.5f}, pur: {pur:.5f}')
        self.model.train()
        # print()        

    def evaluate_unique_cluster_views(self, y_pred_k, centers, features):
        '''
        评估各个视角的聚类结果
        '''
        # y_pred_k, centers, features = self.view_sp_cluster()
        y_pred_list = []
        q_list = [[] for i in range(self.views)]
        y_true = self.dataset.y.cpu().numpy()
        # self.init_sp_cluster_centers(centers)
        self.model.eval()
        with torch.no_grad():
            for view in range(self.views):
                for i in range(0, len(self.dataset), 1024):
                    features_batch = features[view][i:i+1024]
                    q = self.model.cluster_layers[view](features_batch)
                    q = enhance_distribution(q)
                    q_list[view].append(q)
        q_stacked = [torch.cat(f, dim = 0) for f in q_list]
        q_stacked = torch.stack(q_stacked, dim=0)   # [v, N, K]
        avg_q = q_stacked.mean(dim = 0)
        y_pred = torch.argmax(avg_q, dim = 1)
        y_pred = y_pred.cpu().numpy()
        acc = Nmetrics.acc(y_true, y_pred)
        nmi = Nmetrics.nmi(y_true, y_pred)
        ari = Nmetrics.ari(y_true, y_pred)
        pur = Nmetrics.pur(y_true, y_pred)
        print(f'Unique: acc: {acc:.5f}, nmi: {nmi:.5f}, ari: {ari:.5f}, pur: {pur:.5f}')
        self.model.train()
        indices = {
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'pur': pur
        }
        return indices        

    def evaluate_unique_cluster(self, y_pred_k, center, features):
        '''
        评估各个视角的聚类结果
        '''
        # y_pred_k, center, features = self.unique_cluster()
        # center = torch.tensor(center).to(self.device)
        y_pred_k = y_pred_k[0].cpu().numpy()
        y_pred_list = []
        y_true = self.dataset.y.cpu().numpy()
        # self.model.unique_center = center
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(self.dataset), 1024):
                features_batch = features[i:i+1024]
                q = student_distribution(features_batch, self.model.unique_center)
                q = enhance_distribution(q)
                y_pred_list.append(torch.argmax(q, dim = 1))
        y_pred = torch.cat(y_pred_list, dim = 0).cpu().numpy()
        acc = Nmetrics.acc(y_true, y_pred)
        nmi = Nmetrics.nmi(y_true, y_pred)
        ari = Nmetrics.ari(y_true, y_pred)
        pur = Nmetrics.pur(y_true, y_pred)
        # sil = silhouette_score(features.cpu().numpy(), y_pred)
        print(f'Unique: acc: {acc:.5f}, nmi: {nmi:.5f}, ari: {ari:.5f}, pur: {pur:.5f}')
        self.model.train()
        indices = {
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'pur': pur,
        }
        return indices

    def train(self):
        self.model.eval()
        with torch.no_grad():
            y_pred_sp, centers_sp, features_sp =  self.view_sp_cluster()
            y_pred_uq, centers_uq, features_uq = self.unique_cluster()
            self.model.unique_center = centers_uq
            cluster_unique_assign = student_distribution(features_uq, self.model.unique_center)
            cluster_unique_assign = enhance_distribution(cluster_unique_assign)
            # 更新公共的聚类质心和每个视角的聚类质心
            self.init_sp_cluster_centers(centers_sp)
            self.evaluate_sp_cluster(y_pred_sp, centers_sp, features_sp)    
            new_indices = self.evaluate_unique_cluster(y_pred_uq, centers_uq, features_uq)
            # new_indices = self.evaluate_unique_cluster_views(y_pred_sp, centers_sp, features_sp)
            is_updated =  self.model.update_best_indice(new_indices)
            # print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
            #             (self.model.best_indice['acc'], self.model.best_indice['nmi'],self.model.best_indice['ari'],self.model.best_indice['pur']))
            if is_updated is True and self.args.save is True:
                print('saving model to:', self.args.weights)
                self.model.save_model()


        for i in range(self.args.epochs):
            print(f'epoch: {i + 1}')
            self.model.train()
            # 评估一下预训练的结果，同时初始化质心，评估不涉及到后续的训练
            # 评估一下全局的聚类结果，评估不涉及到后续的训练
            # 更新自监督聚类标签
            losses_sum = [0.0] * self.views
            for x, y, idx in self.data_loader:
                x = [xi.to(self.device) for xi in x]
                self.opt.zero_grad()    
                features, reconstructed_x, cluster_sp_assign = self.model(x, is_pretrain = False)
                if self.args.normalize == 1:
                    features = [F.normalize(f, p = 2, dim = 1) for f in features]
                z = torch.stack(features, dim=0).mean(0)
                reconstructed_z = [self.model.generator[view](z) for view in range(self.views)]
                # reconstructed_z = self.model.generator(z)
                #with torch.no_grad():
                losses = self.loss_fn(
                    x = x,
                    z = z,
                    features = features,
                    reconstructed_x = reconstructed_x,
                    reconstructed_z = reconstructed_z,
                    cluster_unique_assign = cluster_unique_assign[idx], 
                    cluster_sp_assign = cluster_sp_assign,
                    args = self.args
                )
                loss = sum(losses) / self.views
                for view in range(self.views):
                    losses_sum[view] += losses[view]
                loss.backward()
                self.opt.step()
            
            loss_total = sum(losses_sum) / self.views
            losses_sum = [loss_total] + losses_sum
            for view in range(len(losses_sum)):
                losses_sum[view] = losses_sum[view].item() / len(self.data_loader)
            print(f'loss: {losses_sum}')

            # 更新公共的聚类质心和每个视角的聚类质心，并且评估结果
            if (i + 1) % self.args.update_interval == 0:
                print('更新聚类质心')
                self.model.eval()
                with torch.no_grad():
                    # y_pred_sp, centers_sp, features_sp =  self.view_sp_cluster()
                    y_pred_uq, centers_uq, features_uq = self.unique_cluster()
                    # self.model.unique_center = self.args.m * self.model.unique_center + (1 - self.args.m) * centers_uq
                    self.model.unique_center = centers_uq
                    cluster_unique_assign = student_distribution(features_uq, self.model.unique_center)
                    cluster_unique_assign = enhance_distribution(cluster_unique_assign)
                    # 更新公共的聚类质心和每个视角的聚类质心
                    
                    # self.update_sp_cluster_centers(centers_sp)
                    # self.evaluate_sp_cluster(y_pred_sp, centers_sp, features_sp)    
                    new_indices = self.evaluate_unique_cluster(y_pred_uq, centers_uq, features_uq)
                    # new_indices = self.evaluate_unique_cluster_views(y_pred_sp, centers_sp, features_sp)
                    is_updated =  self.model.update_best_indice(new_indices)
                    # print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                    #     (self.model.best_indice['acc'], self.model.best_indice['nmi'],self.model.best_indice['ari'],self.model.best_indice['pur']))
                    if is_updated is True and self.args.save is True:
                        print('saving model to:', self.args.weights)
                        self.model.save_model()
            
            # 评估聚类结果
            elif (i + 1) % self.args.cluster_interval == 0 and i != 0:
                self.model.eval()
                with torch.no_grad():
                    # y_pred_sp, centers_sp, features_sp =  self.view_sp_cluster()
                    y_pred_uq, centers_uq, features_uq = self.unique_cluster()
                    self.evaluate_sp_cluster(y_pred_sp, centers_sp, features_sp)    
                    # new_indices = self.evaluate_unique_cluster(y_pred_uq, centers_uq, features_uq)
                    # new_indices = self.evaluate_unique_cluster_views(y_pred_sp, centers_sp, features_sp)
                    is_updated =  self.model.update_best_indice(new_indices)
                    # print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                    #     (self.model.best_indice['acc'], self.model.best_indice['nmi'],self.model.best_indice['ari'],self.model.best_indice['pur']))
                    if is_updated is True and self.args.save is True:
                        print('saving model to:', self.args.weights)
                        self.model.save_model()
            # self.scheduler.step()
            print()
    
    def test(self):
        self.model = self.model.load_model(self.device)
        self.model.eval()
        y_pred_uq, centers_uq, features_uq = self.unique_cluster()
        self.model.unique_center = centers_uq
        cluster_unique_assign = student_distribution(features_uq, self.model.unique_center)
        cluster_unique_assign = enhance_distribution(cluster_unique_assign)
        self.evaluate_unique_cluster(y_pred_uq, centers_uq, features_uq)
        fig_dir = os.path.join(self.args.save_dir, self.args.dataset + '.pdf')
        plot_tsne(features_uq.cpu().numpy(), y_pred_uq[0].cpu().numpy(), fig_dir, self.args.seed)
        is_pause = 1