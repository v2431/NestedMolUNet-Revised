# -*- coding: utf-8 -*-
"""
databuild_dti.py
Created on Wed Mar 15 16:56:49 2023

@author: Fanding Xu
"""

import pandas as pd
import os
import pickle
import torch
import numpy as np
import random
from tqdm import tqdm
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import logging
from transformers import EsmTokenizer

def worker_init_fn(worker_id):
    """确保DataLoader的多进程也能复现"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


"""
The following funcs are copied and adaptedfrom https://github.com/thinng/GraphDTA
"""
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
# seq_voc = "ABDEFGHJKLMNPQRSTUVW"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}

def seq_cat(prot, max_seq_len):
    x = torch.zeros(max_seq_len, dtype=torch.int32)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)  # 未知字符默认为 0
    return x  

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length, dtype=np.int64)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
            encoding[idx] = 0  # 明确设置为0
    return encoding

# ESM tokenizer 处理蛋白质序列
def esm_tokenize_protein(sequence, tokenizer, max_length=1200):
    """
    Use ESM tokenizer to encode protein sequence.
    Args:
        sequence (str): Protein string sequence.
        tokenizer: ESM tokenizer instance.
        max_length: Maximum encoding length of input protein string.
    Returns:
        dict: {
            'input_ids': tensor of shape [L],
            'attention_mask': tensor of shape [L]
        }
    """
    # 根据 max_length 参数决定 padding 策略
    if max_length is None:
        tokens = tokenizer(sequence, 
                          return_tensors='pt', 
                          truncation=True,
                          padding=False)
    else:
        tokens = tokenizer(sequence, 
                          return_tensors='pt', 
                          truncation=True,
                          max_length=max_length,
                          padding='max_length')
    return {
        'input_ids': tokens['input_ids'].squeeze(0),
        'attention_mask': tokens['attention_mask'].squeeze(0)
    }

def encode_seq_deepdta(id_seq_dict, max_seq_len=1200):
    target_embd_dict = {}
    for k, v in tqdm(id_seq_dict.items(), desc='Init protein embeds...'):
        seq = seq_cat(v, max_seq_len)
        target_embd_dict[k] = seq
        
    return target_embd_dict

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(filename, " successfully loaded")
    return obj

def df_reg2cls(df, threshold=6):
    # chembl ratio 0.772
    df.loc[df.ec50 < threshold, 'ec50']=0
    df.loc[df.ec50 >= threshold, 'ec50']=1
    df['ec50'] = df['ec50'].apply(int)
    return df
    



class DTIDataLoader(DataLoader):
    def __init__(self, data_list, **kwargs):
        super().__init__(data_list, collate_fn=self.collate_fn, **kwargs)
        
    def collate_fn(self, batch):
        mols = []
        tars = []
        ys = []
        for mol, tar, y in batch:
            mols.append(mol)
            tars.append(tar)
            ys.append(y)
        mols = Batch.from_data_list(mols)
        
        # 检查 tars 是否为 dict 格式
        if isinstance(tars[0], dict):
            # ===== 🔑 CRITICAL: 检查 dict keys 一致性 =====
            keys0 = set(tars[0].keys())
            for i, t in enumerate(tars[1:], start=1):
                if set(t.keys()) != keys0:
                    raise ValueError(f"Inconsistent protein dict keys in batch: idx0={keys0}, idx{i}={set(t.keys())}")
            
            # ===== 🔑 CRITICAL: 检查 tensor shape 一致性 =====
            if 'features' in tars[0]:
                shapes = [tuple(t['features'].shape) for t in tars]
                if len(set(shapes)) != 1:
                    raise ValueError(f"Inconsistent feature shapes in batch: {set(shapes)}")
            if 'input_ids' in tars[0]:
                shapes = [tuple(t['input_ids'].shape) for t in tars]
                if len(set(shapes)) != 1:
                    raise ValueError(f"Inconsistent input_ids shapes in batch: {set(shapes)}")
            if 'tokens' in tars[0]:
                shapes = [tuple(t['tokens'].shape) for t in tars]
                if len(set(shapes)) != 1:
                    raise ValueError(f"Inconsistent tokens shapes in batch: {set(shapes)}")
            
            if 'input_ids' in tars[0]:
                # ESM token 模式
                tars_ids = torch.stack([t['input_ids'] for t in tars], dim=0).long()
                tars_mask = torch.stack([t['attention_mask'] for t in tars], dim=0).long()
                tars = {'input_ids': tars_ids, 'attention_mask': tars_mask}
            elif 'tokens' in tars[0]:
                # CNN token 模式：现在CNN也需要attention_mask了
                tars_ids = torch.stack([t['tokens'] for t in tars], dim=0).long()
                tars_mask = torch.stack([t['attention_mask'] for t in tars], dim=0).long()
                tars = {'tokens': tars_ids, 'attention_mask': tars_mask}
            elif 'features' in tars[0]:
                # 预计算特征模式
                tars_features = torch.stack([t['features'] for t in tars], dim=0).float()
                tars_mask = torch.stack([t['attention_mask'] for t in tars], dim=0).long()
                tars = {'features': tars_features, 'attention_mask': tars_mask}
        else:
            # 传统格式，直接 stack
            tars = torch.stack(tars, dim=0)
        
        # 防御性处理 ys（可能是 Python 数或 tensor）
        if torch.is_tensor(ys[0]):
            ys = torch.stack([y.view(-1) for y in ys], dim=0).float().squeeze(-1)
        else:
            ys = torch.tensor(ys, dtype=torch.float32)
        return (mols, tars, ys)
    
def get_loader_random(dataset, batch_size=64, seed=114514, num_workers=0):
    data_tr, data_te = train_test_split(dataset, test_size=0.2, random_state=seed)
    data_te, data_va = train_test_split(data_te, test_size=0.5, random_state=seed)
    torch.manual_seed(seed=seed)
    # 为DataLoader创建固定的随机种子生成器
    g = torch.Generator().manual_seed(seed)
    loader_tr = DTIDataLoader(data_tr, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, generator=g,
                               worker_init_fn=worker_init_fn,
                               multiprocessing_context='spawn' if num_workers > 0 else None,
                               persistent_workers=True if num_workers > 0 else False)
    loader_va = DTIDataLoader(data_va, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers,
                               worker_init_fn=worker_init_fn,
                               multiprocessing_context='spawn' if num_workers > 0 else None)
    loader_te = DTIDataLoader(data_te, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers,
                               worker_init_fn=worker_init_fn,
                               multiprocessing_context='spawn' if num_workers > 0 else None)
    return loader_tr, loader_va, loader_te

class easy_data:
    def __init__(self, datamaker, get_fp=False, with_hydrogen=False, with_coordinate=False,
                  seed=123):
        self.datamaker = datamaker
        self.get_fp = get_fp
        self.with_hydrogen = with_hydrogen
        self.with_coordinate = with_coordinate
        self.seed = seed
        self.rdkit_fp = False
        self.mask_attr = False

    def process(self, smi, mol):
        data = self.datamaker(smi, mol, get_fp=self.get_fp,
                              with_hydrogen=self.with_hydrogen, with_coordinate=self.with_coordinate, seed=self.seed)
        return data
    
    def __repr__(self):
        info = f"easy_data(rdkit_fp={self.rdkit_fp}, get_fp={self.get_fp}, mask_attr={self.mask_attr})"
        return info
######################
# from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
# from torch_geometric.utils import from_dgl, sort_edge_index
# class easy_data:
#     def __init__(self, datamaker, get_fp=False, with_hydrogen=False, with_coordinate=False, node_featurizer=CanonicalAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(self_loop=False),
#                   seed=123):
#         self.datamaker = datamaker
#         self.node_featurizer = node_featurizer
#         self.edge_featurizer = edge_featurizer
#         self.get_fp = get_fp
#         self.with_hydrogen = with_hydrogen
#         self.with_coordinate = with_coordinate
#         self.seed = seed
#         self.rdkit_fp = False
#         self.mask_attr = False

#     def process(self, smi, mol):
#         g = smiles_to_bigraph(smi, add_self_loop=False, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer)
#         g.ndata['x'] = g.ndata.pop('h')
#         g.edata['edge_attr'] = g.edata.pop('e')
#         data = from_dgl(g)
#         data.edge_index, data.edge_attr = sort_edge_index(data.edge_index.to(torch.long), data.edge_attr.to(torch.float32))
#         data.x = data.x.to(torch.float32)
#         data.smiles = smi
#         return data
    
#     def __repr__(self):
#         info = f"easy_data(rdkit_fp={self.rdkit_fp}, fpSize={self.fpSize}, mask_attr={self.mask_attr})"
#         return info
######################

def prepare_dataset(datamaker, num_processor, dataset='bindingdb', use_esm=False, max_seq_len=None):
    root = f'./dataset/data/DTI/{dataset}'
    full = pd.read_csv(os.path.join(root, 'full.csv'))
    # 保持顺序去重
    smiles = list(dict.fromkeys(full['SMILES'].tolist()))
    protein = list(dict.fromkeys(full['Protein'].tolist()))
    
    # 先把 SMILES 转成 mol，并过滤掉无效的
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid_pairs = [(s, m) for s, m in zip(smiles, mols) if m is not None]
    valid_smiles = [s for s, m in valid_pairs]
    print(f"Filtered {len(smiles) - len(valid_smiles)} invalid SMILES, remaining: {len(valid_smiles)}")
    
    ed = easy_data(datamaker, get_fp=False, with_hydrogen=False, with_coordinate=False, seed=123)
    data_list = list(Parallel(n_jobs=num_processor)(delayed(ed.process)(smi, mol)
                                         for smi, mol in tqdm(valid_pairs,
                                                             total=len(valid_pairs),
                                                             desc='Generating drug data...')))
    drug_dict = {}
    for data in data_list: drug_dict[data.smiles] = data
    
    prot_dict = {}
    
    if use_esm:
        # ===== 🔑 CRITICAL: ESM token 模式需要固定长度才能 stack =====
        if max_seq_len is None:
            raise ValueError("ESM token mode requires fixed max_seq_len for stacking in collate")
        
        # 使用ESM tokenizer
        from transformers import EsmTokenizer
        
        # 直接使用构造函数加载tokenizer，完全避开from_pretrained的路径验证
        model_dir = './dataset/data/esm2_t30_150M_UR50D/'
        tokenizer = EsmTokenizer(vocab_file=os.path.join(model_dir, 'vocab.txt'))
        
        # max_seq_len 已经在上面检查过不为 None
        for seq in tqdm(protein, desc='seq2esm_tokens...'):
            tokens = esm_tokenize_protein(seq, tokenizer, max_length=max_seq_len)
            # tokens 现在是 dict: {'input_ids': ..., 'attention_mask': ...}
            prot_dict[seq] = tokens  # 直接存储 dict
        
        torch.save(drug_dict, os.path.join(root, 'drugs.pth'))
        torch.save(prot_dict, os.path.join(root, 'protein_esm.pth'))
    else:
        # 使用传统整数编码
        for seq in tqdm(protein, desc='seq2embd...'):
            # CNN模式下不限制max_seq_len
            ml = max_seq_len if max_seq_len is not None else 1200
            prot_dict[seq] = torch.from_numpy(integer_label_protein(seq, ml))
        
        torch.save(drug_dict, os.path.join(root, 'drugs.pth'))
        torch.save(prot_dict, os.path.join(root, 'protein.pth'))

def precompute_protein_features(dataset='bindingdb', esm_model='esm2_t30_150M_UR50D', max_seq_len=1200, finetune=False, device=0):
    """
    提前计算蛋白质特征并保存
    
    Args:
        dataset: 数据集名称
        esm_model: ESM模型名称
        max_seq_len: 蛋白质序列最大长度
        finetune: 微调策略
        device: CUDA设备ID
    """
    import torch
    import numpy as np
    from tqdm import tqdm
    from transformers import EsmTokenizer
    
    # 设置随机种子，确保特征计算可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # 启用确定性计算
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    root = f'./dataset/data/DTI/{dataset}'
    full = pd.read_csv(os.path.join(root, 'full.csv'))
    # 保持顺序去重
    protein_seqs = list(dict.fromkeys(full['Protein'].tolist()))
    
    # 加载tokenizer
    model_dir = f'./dataset/data/{esm_model}/'
    tokenizer = EsmTokenizer(vocab_file=os.path.join(model_dir, 'vocab.txt'))
    
    # 加载ESM模型
    from models.model_dti import ProteinESM
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 不指定embedding_dim，让ProteinESM自动设置
    protein_extractor = ProteinESM(
        embedding_dim=None,  # 自动根据模型大小设置
        pretrained_model=model_dir, 
        finetune=finetune
    )
    protein_extractor.to(device)
    protein_extractor.eval()
    
    # 验证模型模式
    print(f"🔍 ProteinESM training mode: {protein_extractor.training}")
    if hasattr(protein_extractor, 'esm'):
        print(f"🔍 ESM model training mode: {protein_extractor.esm.training}")
    
    # 获取自动设置的proj_dim
    proj_dim = protein_extractor.proj_dim
    print(f"🔍 Using proj_dim: {proj_dim} for feature extraction")
    
    # 提前计算特征
    prot_features_dict = {}
    with torch.no_grad():
        for seq in tqdm(protein_seqs, desc='Precomputing protein features...'):
            # 生成tokens - 使用 max_seq_len 作为 max_length，确保与训练时的处理一致
            tokens = esm_tokenize_protein(seq, tokenizer, max_length=max_seq_len)
            # tokens 现在是 dict: {'input_ids': ..., 'attention_mask': ...}
            input_ids = tokens['input_ids'].unsqueeze(0).to(device)  # [1, L]
            attn_mask = tokens['attention_mask'].unsqueeze(0).to(device)  # [1, L]
            
            # 计算特征
            features = protein_extractor(input_ids, attention_mask=attn_mask)  # [1, L, proj_dim]
            features = features.squeeze(0).cpu()  # [L, proj_dim]
            
            # ===== 🔑 CRITICAL: 保存时就 pad/truncate 到 max_seq_len =====
            L0 = features.size(0)
            if L0 > max_seq_len:
                features = features[:max_seq_len]
                attn_mask_cpu = torch.ones(max_seq_len, dtype=torch.long)
            elif L0 < max_seq_len:
                pad = torch.zeros(max_seq_len - L0, features.size(1), dtype=features.dtype)
                features = torch.cat([features, pad], dim=0)
                attn_mask_cpu = torch.zeros(max_seq_len, dtype=torch.long)
                attn_mask_cpu[:L0] = 1
            else:
                attn_mask_cpu = tokens['attention_mask'].cpu()
            
            # 保存 features + attention_mask
            prot_features_dict[seq] = {
                'features': features.float(),
                'attention_mask': attn_mask_cpu
            }
    
    # 保存特征
    features_path = os.path.join(root, f'protein_esm_features_{esm_model}.pth')
    torch.save(prot_features_dict, features_path)
    print(f"Protein features saved to: {features_path}")
    return features_path


class BenchmarkDataset(Dataset):
    def __init__(self, df_path, drug_dict, prot_dict, max_seq_len=1200):
        self.df = pd.read_csv(df_path)        
        self.drug_dict = drug_dict
        self.prot_dict = prot_dict
        self.max_seq_len = max_seq_len
        
        # 验证所有蛋白质序列是否存在于字典中
        missing_proteins = []
        for _, row in self.df.iterrows():
            if row['Protein'] not in self.prot_dict:
                missing_proteins.append(row['Protein'])
        if missing_proteins:
            print(f"WARNING: {len(missing_proteins)} proteins missing from prot_dict")
            print(f"First 5 missing proteins: {missing_proteins[:5]}")
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        smiles = row['SMILES']
        protein_seq = row['Protein']
        
        # 确保 SMILES 存在
        if smiles not in self.drug_dict:
            raise KeyError(f"SMILES not found: {smiles[:50]}...")
        
        # 确保蛋白质序列存在
        if protein_seq not in self.prot_dict:
            raise KeyError(f"Protein sequence not found: {protein_seq[:50]}...")
        
        drug = self.drug_dict[smiles]
        prot = self.prot_dict[protein_seq]
        y = row['Y']
        
        # ===== 构造 protein 输入（统一成：要么 dict，要么 tensor）=====
        if isinstance(prot, dict) and 'input_ids' in prot:
            # ESM token 模式：直接用 tokenizer 产物
            prot_dict_item = {
                'input_ids': prot['input_ids'],
                'attention_mask': prot['attention_mask'].long()
            }

        elif isinstance(prot, dict) and 'features' in prot:
            # ===== 🔑 预计算特征 dict 模式（新格式：已 pad + 带 mask）=====
            feat = prot['features']
            mask = prot.get('attention_mask', None)
            
            # 形状检查
            if feat.dim() != 2 or feat.size(1) <= 0:
                raise ValueError(f"Invalid precomputed feature shape: {feat.shape}, expected [L, D] with D > 0")
            
            # 如果已有 mask 且长度正确，直接使用
            if mask is not None and mask.size(0) == self.max_seq_len:
                prot_dict_item = {
                    'features': feat.float(),
                    'attention_mask': mask.long()
                }
            else:
                # 兼容旧格式：基于 feature 长度构造 mask
                L0 = feat.size(0)
                if L0 != self.max_seq_len:
                    # 需要 pad/truncate
                    if L0 < self.max_seq_len:
                        pad = torch.zeros(self.max_seq_len - L0, feat.size(1), dtype=feat.dtype)
                        feat = torch.cat([feat, pad], dim=0)
                    else:
                        feat = feat[:self.max_seq_len]
                    L0 = self.max_seq_len
                
                # 如果有 mask 但长度不对，重新构造
                if mask is not None:
                    L = min(mask.sum().item(), L0)
                else:
                    L = L0
                
                prot_attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
                prot_attention_mask[:L] = 1
                prot_dict_item = {
                    'features': feat.float(),
                    'attention_mask': prot_attention_mask
                }

        elif torch.is_tensor(prot) and torch.is_floating_point(prot):
            # ✅ 预计算特征真实情况：prot_dict[seq] 是 Tensor [L, D]
            feat = prot
            # ===== 🔑 CRITICAL: 严格的形状检查 =====
            if feat.dim() != 2 or feat.size(1) <= 0:
                raise ValueError(f"Invalid precomputed feature shape: {feat.shape}, expected [L, D] with D > 0")
            
            # ===== 🔑 CRITICAL: 基于 feature 真实长度构造 mask =====
            L0 = feat.size(0)
            L = min(L0, self.max_seq_len)
            
            # pad / truncate 到 max_seq_len
            if L0 < self.max_seq_len:
                pad = torch.zeros(self.max_seq_len - L0, feat.size(1), dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=0)
            elif L0 > self.max_seq_len:
                feat = feat[:self.max_seq_len]
            
            prot_attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            prot_attention_mask[:L] = 1
            prot_dict_item = {
                'features': feat.float(),
                'attention_mask': prot_attention_mask
            }

        else:
            # CNN 整数编码：prot 是 [L] long（padding 为 0）
            prot_tensor = prot.long() if torch.is_tensor(prot) else torch.tensor(prot, dtype=torch.long)
            
            # ===== 🔑 小加固：显式确保是 1D =====
            prot_tensor = prot_tensor.view(-1)
            
            # ===== 🔑 CRITICAL: 显式截断/补齐到 max_seq_len =====
            if prot_tensor.numel() > self.max_seq_len:
                prot_tensor = prot_tensor[:self.max_seq_len]
            elif prot_tensor.numel() < self.max_seq_len:
                pad = torch.zeros(self.max_seq_len - prot_tensor.numel(), dtype=prot_tensor.dtype)
                prot_tensor = torch.cat([prot_tensor, pad], dim=0)
            
            # ✅ 关键：mask 由非零 token 决定（不是用 numel() 推长度）
            prot_attention_mask = (prot_tensor != 0).to(dtype=torch.long)
            
            prot_dict_item = {
                'tokens': prot_tensor,
                'attention_mask': prot_attention_mask
            }
        if torch.is_tensor(drug.x) and torch.is_floating_point(drug.x) and torch.isnan(drug.x).any():
            drug.x = torch.nan_to_num(drug.x, nan=0.0)
            print(f"Warning: Drug {smiles[:50]} contains NaN values, replaced with 0")
        
        # 检查标签是否为NaN (使用 pd.isna 覆盖标量/字符串/None 等情况)
        if pd.isna(y):
            y = 0.0
            print(f"Warning: Label contains NaN values, replaced with 0")
        else:
            try:
                y = float(y)
                if not np.isfinite(y):
                    y = 0.0
                    print(f"Warning: Label contains non-finite values, replaced with 0")
            except (TypeError, ValueError):
                y = 0.0
                print(f"Warning: Label could not be converted to float, replaced with 0")
        
        return (drug, prot_dict_item, y)
    
    
def get_benchmark_loader(task='bindingdb', split='random',
                         batch_size=64, seed=114514, num_workers=0, use_esm=False, use_precomputed_features=False, esm_model='esm2_t30_150M_UR50D', features_suffix='', max_seq_len=1200):
    # DDP 初始化检查
    import torch.distributed as dist
    dist_inited = dist.is_available() and dist.is_initialized()
    is_rank0 = (not dist_inited) or dist.get_rank() == 0
    
    root = f'dataset/data/DTI/{task}/{split}'
    if is_rank0:
        print(f"Loading drug dictionary from: dataset/data/DTI/{task}/drugs.pth")
    drug_dict = torch.load(f'dataset/data/DTI/{task}/drugs.pth', weights_only=False)
    if is_rank0:
        print(f"Loaded drug dictionary with {len(drug_dict)} entries")
    
    # 辅助函数：优先读取 filtered 版本
    def get_csv_path(root, split_name):
        filtered_path = os.path.join(root, f'{split_name}.filtered.csv')
        normal_path = os.path.join(root, f'{split_name}.csv')
        if os.path.exists(filtered_path):
            return filtered_path
        return normal_path
    
    # 过滤 split CSV 中的 invalid SMILES，总是写入 filtered 文件
    valid_smiles_set = set(drug_dict.keys())
    for split_name in ['train', 'val', 'test']:
        split_path = os.path.join(root, f'{split_name}.csv')
        filtered_path = os.path.join(root, f'{split_name}.filtered.csv')
        
        if os.path.exists(split_path):
            df = pd.read_csv(split_path)
            df_filtered = df[df['SMILES'].isin(valid_smiles_set)].reset_index(drop=True)
            
            if is_rank0:
                dropped = len(df) - len(df_filtered)
                if dropped > 0:
                    print(f"Filtered {dropped} invalid SMILES from {split_name}.csv")
                # 原子写入：先写临时文件再替换
                tmp_path = filtered_path + f".tmp_{os.getpid()}_rank0"
                df_filtered.to_csv(tmp_path, index=False)
                os.replace(tmp_path, filtered_path)
        else:
            # 原始 split 不存在时，删除旧的 filtered 文件
            if is_rank0 and os.path.exists(filtered_path):
                os.remove(filtered_path)
        
        if dist_inited:
            dist.barrier()  # 每个 split 都 barrier 一次，确保同步
    
    if use_precomputed_features:
        # 加载预计算的特征
        # 每个 rank 都自己加载预计算特征（共享存储下最稳的方式）
        base_dir = f'dataset/data/DTI/{task}'
        if features_suffix:
            features_path = os.path.join(base_dir, f'protein_esm_features_{esm_model}_{features_suffix}.pth')
        else:
            features_path = os.path.join(base_dir, f'protein_esm_features_{esm_model}.pth')
        
        # 检查文件是否存在
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Precomputed protein features not found at: {features_path}")
        
        # barrier 同步
        if dist_inited:
            dist.barrier()
        
        if is_rank0:
            print(f"Loading precomputed protein features from: {features_path}")
        prot_dict = torch.load(features_path, weights_only=False)
        if is_rank0:
            print(f"Loaded precomputed protein features from: {features_path}")
            print(f"Number of precomputed protein features: {len(prot_dict)}")
        
        # barrier 同步
        if dist_inited:
            dist.barrier()
        
        # 打印第一个特征的形状
        if prot_dict and is_rank0:
            first_seq = next(iter(prot_dict.keys()))
            first_feat = prot_dict[first_seq]
            # ===== 🔑 兼容新旧格式 =====
            if isinstance(first_feat, dict):
                print(f"First precomputed feature (NEW format): features {first_feat['features'].shape}, mask {first_feat['attention_mask'].shape}")
            else:
                print(f"First precomputed feature (OLD format): shape {first_feat.shape}, dtype {first_feat.dtype}")
        
        # ===== 🔑 DDP 优化：只让 rank0 做 NaN 检查 =====
        if is_rank0:
            nan_count = 0
            for seq, item in prot_dict.items():
                # ===== 🔑 兼容新旧格式 =====
                if isinstance(item, dict):
                    feat = item['features']
                else:
                    feat = item
                
                if torch.is_tensor(feat) and torch.is_floating_point(feat):
                    if torch.isnan(feat).any():
                        nan_count += 1
                        if isinstance(item, dict):
                            item['features'] = torch.where(torch.isnan(feat), torch.zeros_like(feat), feat)
                        else:
                            prot_dict[seq] = torch.where(torch.isnan(feat), torch.zeros_like(feat), feat)
            if nan_count > 0:
                print(f"Found {nan_count} proteins with NaN values, replaced with 0")
            else:
                print("All proteins have valid values (no NaN found)")
    elif use_esm:
        prot_dict = torch.load(f'dataset/data/DTI/{task}/protein_esm.pth', weights_only=False) 
        if is_rank0:
            print(f"Number of protein ESM tokens: {len(prot_dict)}")
        
        # ===== 🔑 DDP 优化：只让 rank0 做 NaN 检查 =====
        if is_rank0:
            nan_count = 0
            for seq, item in prot_dict.items():
                if not isinstance(item, dict):
                    continue
                for k in ['input_ids', 'attention_mask']:
                    if k in item and torch.is_tensor(item[k]):
                        if torch.is_floating_point(item[k]) and torch.isnan(item[k]).any():
                            nan_count += 1
                            item[k] = torch.nan_to_num(item[k], nan=0.0)
            if nan_count > 0:
                print(f"Found {nan_count} proteins with NaN values (fixed to 0)")
            else:
                print("All proteins have valid values (no NaN found)")
    else:
        # 加载蛋白质的整数编码（用于CNN模型）
        prot_dict = torch.load(f'dataset/data/DTI/{task}/protein.pth', weights_only=False)                     
        if is_rank0:
            print(f"Number of protein tokens: {len(prot_dict)}")
        # 确保蛋白质特征是整数类型（兼容旧 float32 格式的 protein.pth）
        converted = 0
        for seq in prot_dict:
            if prot_dict[seq].dtype == torch.float32:
                prot_dict[seq] = prot_dict[seq].long()
                converted += 1
        if is_rank0:
            if converted > 0:
                print(f"Converted {converted} proteins from float32 to long (legacy format)")
            else:
                print("All protein tokens are already integer type")
        
        # 检查NaN值 (CNN 模式下是 float tensor)
        nan_count = 0
        for seq, feat in prot_dict.items():
            if torch.is_tensor(feat) and torch.is_floating_point(feat):
                if torch.isnan(feat).any():
                    nan_count += 1
        if nan_count > 0 and is_rank0:
            print(f"Found {nan_count} proteins with NaN values")
    
    # 验证预计算特征字典 (移到三个分支之后)
    if use_precomputed_features and prot_dict is not None:
        # 检查训练集、验证集和测试集中的所有蛋白质序列是否都在预计算特征字典中
        for split_name in ['train', 'test', 'val']:
            csv_path = get_csv_path(root, split_name)
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            missing_proteins = []
            for _, row in df.iterrows():
                if row['Protein'] not in prot_dict:
                    missing_proteins.append(row['Protein'])
            if missing_proteins:
                if is_rank0:
                    print(f"WARNING: {len(missing_proteins)} proteins missing from precomputed features in {split_name} set")
                    print(f"First 5 missing proteins: {missing_proteins[:5]}")
            else:
                if is_rank0:
                    print(f"All proteins found in precomputed features for {split_name} set")
    
    # 检查 split 文件是否存在
    for split_name in ['train', 'val', 'test']:
        csv_path = get_csv_path(root, split_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing split file: {csv_path}")
    
    train_dataset = BenchmarkDataset(get_csv_path(root, 'train'), drug_dict, prot_dict, max_seq_len)
    test_dataset = BenchmarkDataset(get_csv_path(root, 'test'), drug_dict, prot_dict, max_seq_len)
    val_dataset = BenchmarkDataset(get_csv_path(root, 'val'), drug_dict, prot_dict, max_seq_len)
    
    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    loader_tr = DTIDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True,
                               worker_init_fn=worker_init_fn,
                               multiprocessing_context='spawn' if num_workers > 0 else None,
                               persistent_workers=True if num_workers > 0 else False)
    loader_te = DTIDataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, drop_last=False,
                               worker_init_fn=worker_init_fn,
                               multiprocessing_context='spawn' if num_workers > 0 else None)
    loader_va = DTIDataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, drop_last=False,
                               worker_init_fn=worker_init_fn,
                               multiprocessing_context='spawn' if num_workers > 0 else None)
    return loader_tr, loader_va, loader_te



# ********** ALDH2 筛选 *****************************************************************************
class ScreenDataset(Dataset):
    def __init__(self, df, mol_dict, tar_dict):
        self.df = df
        self.data_list = []
        for idx, row in df.iterrows():
            drug = mol_dict[row['Smiles']]
            target = tar_dict[row['Target ChEMBL ID']]
            label = float(row['class'])
            y = torch.tensor([label], dtype=torch.float32)
            self.data_list.append((drug, target, y))
            if row['Target ChEMBL ID'] == 'CHEMBL1935':
                self.data_list += [(drug, target, y)] * 10
                if label == 1.0:
                    self.data_list += [(drug, target, y)] * 100
       
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
    
class ScreenLoader(DataLoader):
    def __init__(self, data_list, **kwargs):
        super().__init__(data_list, collate_fn=self.collate_fn, **kwargs)
        
    def collate_fn(self, batch):
        drugs = []
        tars = []
        ys = []
        for drug, target, y in batch:
            drugs.append(drug)
            tars.append(target)
            ys.append(y)
        drugs = Batch.from_data_list(drugs)
        tars = torch.stack(tars, dim=0)
        ys = torch.cat(ys, dim=0)
        return (drugs, tars, ys)

def get_screen_loader(batch_size=64, seed=114514, num_workers=0, use_esm=False):
    df = pd.read_csv('dataset/data/DTI/ChemblICEC.csv')
    df = df.dropna()
    print("Loading drugs...")
    drug_dict = torch.load('dataset/data/DTI/screen/drugs.pth')
    print("Loading proteins...")
    
    if use_esm:
        prot_dict = torch.load('dataset/data/DTI/screen/protein_esm.pth')
    else:
        prot_dict = torch.load('dataset/data/DTI/screen/protein.pth')
    
    print("Generating dataset...")
    dataset = ScreenDataset(df, drug_dict, prot_dict)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=seed)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    loader_tr = ScreenLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    loader_te = ScreenLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, drop_last=False)
    loader_va = ScreenLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, drop_last=False)
    return loader_tr, loader_va, loader_te

    
def prepare_screen_dataset(datamaker, num_processor, use_esm=False, max_seq_len=1200):
    path = './dataset/data/DTI/ChemblICEC.csv'
    full = pd.read_csv(path)
    full = full.dropna()
    # 保持顺序去重
    smiles = list(dict.fromkeys(full['Smiles'].tolist()))
    # protein = set(full['Target ChEMBL ID'].tolist())
    mols = [Chem.MolFromSmiles(s) for s in tqdm(smiles, desc='smiles2mol...')]
    valid_pairs = [(s, m) for s, m in zip(smiles, mols) if m is not None]
    valid_smiles = [s for s, m in valid_pairs]
    print(f"Filtered {len(smiles) - len(valid_smiles)} invalid SMILES, remaining: {len(valid_smiles)}")
    
    ed = easy_data(datamaker, get_fp=False, with_hydrogen=False, with_coordinate=False, seed=123)
    data_list = list(Parallel(n_jobs=num_processor)(delayed(ed.process)(smi, mol)
                                         for smi, mol in tqdm(valid_pairs,
                                                             total=len(valid_pairs),
                                                             desc='Generating drug data...')))
    drug_dict = {}
    for data in data_list: drug_dict[data.smiles] = data
    
    with open ('./dataset/data/DTI/SequenceDict.pkl', 'rb') as f:
        prot_id2seq = pickle.load(f)
        
    prot_dict = {}
    
    if use_esm:
        # 使用ESM tokenizer
        from transformers import EsmTokenizer
        import os
        
        # 直接使用构造函数加载tokenizer
        model_dir = './dataset/data/esm2_t30_150M_UR50D/'
        tokenizer = EsmTokenizer(vocab_file=os.path.join(model_dir, 'vocab.txt'))
        
        for chembl_id, seq in tqdm(prot_id2seq.items(), desc='seq2esm_tokens...'):
            tokens = esm_tokenize_protein(seq, tokenizer, max_length=max_seq_len)
            prot_dict[chembl_id] = tokens
        
        torch.save(drug_dict, './dataset/data/DTI/screen/drugs.pth')
        torch.save(prot_dict, './dataset/data/DTI/screen/protein_esm.pth')
    else:
        # 使用传统整数编码
        for chembl_id, seq in tqdm(prot_id2seq.items(), desc='seq2embd...'):
            ml = max_seq_len if max_seq_len is not None else 1200
            prot_dict[chembl_id] = torch.from_numpy(integer_label_protein(seq, ml))
        
        torch.save(drug_dict, './dataset/data/DTI/screen/drugs.pth')
        torch.save(prot_dict, './dataset/data/DTI/screen/protein.pth')

class EC50ScreenLoader(DataLoader):
    def __init__(self, data_list, **kwargs):
        super().__init__(data_list, collate_fn=self.collate_fn, **kwargs)
        
    def collate_fn(self, batch):
        mols = []
        for mol in batch:
            mols.append(mol)
        mols = Batch.from_data_list(mols)
        return mols
# **************************************************************************************************

if __name__ == "__main__":
    
    # ==================== Benchmark ====================
    import argparse
    from databuild import from_smiles
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='human', choices=['bindingdb', 'human', 'biosnap'], 
                            help='Dataset to preprocess.')
    parser.add_argument('-n_p', '--num_processor', type=int, default=8,
                            help='multi processing num processors')
    parser.add_argument('--use_esm', action='store_true',
                            help='Use ESM tokenizer for protein sequences')
    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='Maximum protein sequence length (default: None)')
    args = parser.parse_args()
    
    prepare_dataset(from_smiles, args.num_processor, args.dataset, args.use_esm, args.max_seq_len)
    
    # ===================================================
    
    # prepare_screen_dataset(from_smiles, args.num_processor)
    
    
    
    # from databuild import from_smiles
    # root = './dataset/data/DTI'
    
    # # for dataset in ['ec50s', 'chembl']: 
    # #     # processing mol
    # #     mol_path = os.path.join(root, f'{dataset}_cid2smi.pkl')
    # #     with open(mol_path, 'rb') as f:
    # #         cid2smi = pickle.load(f)
        
    # #     mol_data_dict = {}
    # #     for cid, smiles in tqdm(cid2smi.items(), total=len(cid2smi)):
    # #         mol = Chem.MolFromSmiles(smiles)
    # #         if mol is not None: 
    # #             data = from_smiles(smiles, mol, get_fp=True, with_coordinate=False)
    # #             mol_data_dict[cid] = data
        
    # #     torch.save(mol_data_dict, os.path.join(root, f'{dataset}_mol_dict.pth'))
    
    # test_df = pd.read_csv(os.path.join(root, 'zinc_test.csv'))
    # smiles_list = test_df['smiles'].tolist()
    # data_list = []
    # for smiles in tqdm(smiles_list):
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is not None: 
    #         data = from_smiles(smiles, mol, get_fp=True, with_coordinate=False)
    #         data_list.append(data)

    # torch.save(data_list, os.path.join(root, 'zinc_test_data_list.pth'))
