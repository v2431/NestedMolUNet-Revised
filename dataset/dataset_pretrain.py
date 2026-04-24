# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:35:22 2023

@author: Fanding Xu
"""

import pickle
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(filename, " successfully loaded")
    return obj

    
class ptDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self._indices = None
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]

    def collate_fn(self, batch):
        graph_list = []
        for data in batch:
            graph_list.append(data)
        graph = Batch.from_data_list(graph_list)
        return graph


class ptDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def get_pretrain_loader(data_list, batch_size=64, seed=123, num_workers=0):
    data_tr, data_te = train_test_split(data_list, test_size=0.2, random_state=seed)
    data_te, data_va = train_test_split(data_te, test_size=0.5, random_state=seed)
    # torch.manual_seed(seed=seed)
    loader_tr = ptDataLoader(ptDataset(data_tr), batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)
    loader_va = ptDataLoader(ptDataset(data_va), batch_size=batch_size,
                             num_workers=num_workers)
    loader_te = ptDataLoader(ptDataset(data_te), batch_size=batch_size,
                             num_workers=num_workers)
    return loader_tr, loader_va, loader_te

    
    






































































































