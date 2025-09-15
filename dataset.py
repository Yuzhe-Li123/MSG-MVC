import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from load_data import load_data
import numpy as np

class MultiViewDataset(Dataset):
    def __init__(self, dataset):
        self.x, self.y = load_data(dataset)
        # 转成 float32
        self.x = [torch.as_tensor(xv, dtype=torch.float32) for xv in self.x]
        # 转成无符号整数
        self.y = torch.as_tensor(self.y, dtype=torch.uint8)
        
    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, idx):
        return[self.x[i][idx] for i in range(len(self.x))] , self.y[idx], idx
    
    def get_num_clusters(self):
        return len(np.unique(self.y))

    def get_views(self):
        return [self.x[i].shape[1] for i in range(len(self.x))]


if __name__ == "__main__":
    data = MultiViewDataset(dataset='BDGP')
    print(data.get_views())