# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:35:22 2023

@author: Fanding Xu
"""

import torch
import pickle
import os
import numpy as np
from itertools import compress
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(filename, " successfully loaded")
    return obj

    
class ppDataset(Dataset):
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


class ppDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def get_pp_loader_random(dataset, batch_size=64, folds=5, seed=123):
    file_path = f'data/processed/{dataset}.pkl'
    assert os.path.exists(file_path), "Processed dataset not found, please check the path and run databuild.py first."
    data_list = read_pickle(file_path)
    loaders_tr, loaders_va, loaders_te = [], [], []
    for fold in range(folds):
        data_tr, data_te = train_test_split(data_list, test_size=0.2, random_state=seed+fold)
        data_te, data_va = train_test_split(data_te, test_size=0.5, random_state=seed+fold)
        loaders_tr.append(ppDataLoader(ppDataset(data_tr), batch_size=batch_size))
        loaders_va.append(ppDataLoader(ppDataset(data_va), batch_size=batch_size))
        loaders_te.append(ppDataLoader(ppDataset(data_te), batch_size=batch_size))
    return loaders_tr, loaders_va, loaders_te

def get_pp_loader_random_single(dataset, batch_size=64, seed=123, num_workers=0):
    file_path = f'dataset/data/property/processed/{dataset}.pth'
    assert os.path.exists(file_path), "Processed dataset not found, please check the path and run databuild.py first."
    data_list = torch.load(file_path)
    data_tr, data_te = train_test_split(data_list, test_size=0.2, random_state=seed)
    data_te, data_va = train_test_split(data_te, test_size=0.5, random_state=seed)
    loader_tr = ppDataLoader(ppDataset(data_tr), batch_size=batch_size)
    loader_va = ppDataLoader(ppDataset(data_va), batch_size=batch_size)
    loader_te = ppDataLoader(ppDataset(data_te), batch_size=batch_size)
    return loader_tr, loader_va, loader_te


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def get_pp_loader_scaffold(dataset, batch_size=64, seed=123, num_workers=0):
    load_path = f'dataset/data/property/processed/{dataset}_scaffold_split.pth'
    if os.path.exists(load_path):
        data_tr, data_va, data_te = torch.load(load_path)
        # data_tr, data_va, data_te = read_pickle(load_path)
        print(f'Loaded exist {dataset} split, tr_size={len(data_tr)} va_size={len(data_va)} te_size={len(data_te)}')
    else:
        data_tr, data_va, data_te = scaffold_split(dataset)
        print(f'tr_size={len(data_tr)} va_size={len(data_va)} te_size={len(data_te)}')
        
    # torch.manual_seed(seed) 
    # loader_tr = ppDataLoader(ppDataset(data_tr), batch_size=batch_size,
    #                          shuffle=True, num_workers=num_workers)
    # loader_va = ppDataLoader(ppDataset(data_va), batch_size=batch_size,
    #                          shuffle=False, num_workers=num_workers)
    # loader_te = ppDataLoader(ppDataset(data_te), batch_size=batch_size,
    #                          shuffle=False, num_workers=num_workers)
    
    torch.manual_seed(seed=seed)
    g = torch.Generator()
    loader_tr = ppDataLoader(ppDataset(data_tr), batch_size=batch_size,
                             shuffle=True, drop_last=False, generator=g)
    torch.manual_seed(seed=seed)
    g = torch.Generator()
    loader_va = ppDataLoader(ppDataset(data_va), batch_size=batch_size,
                             shuffle=False, drop_last=False, generator=g)
    torch.manual_seed(seed=seed)
    g = torch.Generator()
    loader_te = ppDataLoader(ppDataset(data_te), batch_size=batch_size,
                             shuffle=False, drop_last=False, generator=g)
    return loader_tr, loader_va, loader_te
    

def get_pp_loader_random_scaffold(dataset, batch_size=64, folds=10, seed=123):
    file_path = f'data/processed/{dataset}.pkl'
    assert os.path.exists(file_path), "Processed dataset not found, please check the path and run databuild.py first."
    data_list = read_pickle(file_path)
    loaders_tr, loaders_va, loaders_te = [], [], []
        
    for fold in range(folds):
        save_smiles_path = f"data/processed/{dataset}_RSP_{fold}.pkl"
        data_tr, data_va, data_te = random_scaffold_split(data_list,
                                                          seed=seed+fold,
                                                          save_smiles_path=save_smiles_path)
        loaders_tr.append(ppDataLoader(ppDataset(data_tr), batch_size=batch_size))
        loaders_va.append(ppDataLoader(ppDataset(data_va), batch_size=batch_size))
        loaders_te.append(ppDataLoader(ppDataset(data_te), batch_size=batch_size))
    
    return  loaders_tr, loaders_va, loaders_te

def scaffold_split(dataset, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   save_smiles=True):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    """
    RDLogger.DisableLog('rdApp.*')

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    file_path = f'dataset/data/property/processed/{dataset}.pth'
    assert os.path.exists(file_path), "Processed dataset not found, please check the path and run databuild.py first."
    data_list = torch.load(file_path)
    print(f"Splitting dataset: {dataset}, with {len(data_list)} samples")
    smiles_list = [data.smiles for data in data_list]
    
    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data[1].y[task_idx].item() for data in data_list])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(data_list)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    
        
    train_data_list = [data_list[i] for i in train_idx]
    valid_data_list = [data_list[i] for i in valid_idx]
    test_data_list = [data_list[i] for i in test_idx]
    
    torch.save((train_data_list, valid_data_list, test_data_list),
               f'dataset/data/property/processed/{dataset}_scaffold_split.pth')
    print(f'splitted smiles saved as data/processed/{dataset}_scaffold_split.pth')
    
    if save_smiles:
        smiles_dict = {}
        smiles_dict['train_smiles'] = [smiles_list[i][1] for i in train_idx]
        smiles_dict['valid_smiles'] = [smiles_list[i][1] for i in valid_idx]
        smiles_dict['test_smiles'] = [smiles_list[i][1] for i in test_idx]
        with open(f'dataset/data/property/processed/{dataset}_scaffold_split_smiles.pkl', 'wb') as f:
            pickle.dump(smiles_dict, f)
        print(f'splitted smiles saved as dataset/data/property/processed/{dataset}_scaffold_split_smiles.pkl')
        
    return train_data_list, valid_data_list, test_data_list


def random_scaffold_split(data_list, task_idx=None, null_value=0,
                          frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0,
                          save_smiles_path: str = None):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """
    RDLogger.DisableLog('rdApp.*')

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    smiles_list = [data.smiles for data in data_list]
    
    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in data_list])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(data_list)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)
    
    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(data_list)))
    n_total_test = int(np.floor(frac_test * len(data_list)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    
    
    train_data_list = [data_list[i] for i in train_idx]
    valid_data_list = [data_list[i] for i in valid_idx]
    test_data_list = [data_list[i] for i in test_idx]
    
    if save_smiles_path is not None:
        smiles_dict = {}
        smiles_dict['train_smiles'] = [smiles_list[i][1] for i in train_idx]
        smiles_dict['valid_smiles'] = [smiles_list[i][1] for i in valid_idx]
        smiles_dict['test_smiles'] = [smiles_list[i][1] for i in test_idx]
        with open(save_smiles_path, 'wb') as f:
            pickle.dump(smiles_dict, f)
        print(f'splitted smiles saved as {save_smiles_path}')
        
    return train_data_list, valid_data_list, test_data_list





































































































