#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提前计算蛋白质特征的脚本
"""

import argparse
import torch
import numpy as np
import random
import os

# 设置随机种子和确定性计算，确保结果可重复
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 启用确定性计算
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 已设置随机种子为 {seed} 并启用确定性计算")

from dataset.databuild_dti import precompute_protein_features

def check_features_exist(dataset, esm_model):
    """检查特定模型的预计算特征是否已存在"""
    features_path = f'./dataset/data/DTI/{dataset}/protein_esm_features_{esm_model}.pth'
    if os.path.exists(features_path):
        return True, features_path
    return False, features_path

if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='bindingdb',
                        help='Dataset name (default: bindingdb)')
    parser.add_argument('-e', '--esm_model', type=str, default='esm2_t12_35M_UR50D',
                        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'],
                        help='ESM model name (default: esm2_t12_35M_UR50D)')
    parser.add_argument('-l', '--max_seq_len', type=int, default=1200,
                        help='Maximum sequence length (default: 1200)')
    parser.add_argument('-f', '--finetune', type=str, default='False',
                        choices=['False', 'True', 'partial'],
                        help='Finetune strategy (default: False - fully frozen)')
    parser.add_argument('-g', '--device', type=int, default=0,
                        help='CUDA device ID to use (default: 0)')
    parser.add_argument('--force', action='store_true',
                        help='Force recomputation even if features already exist')
    args = parser.parse_args()
    
    # 检查特征文件是否已存在
    exists, features_path = check_features_exist(args.dataset, args.esm_model)
    
    if exists and not args.force:
        print(f"\n{'='*70}")
        print(f"特征文件已存在: {features_path}")
        print(f"如需重新计算，请使用 --force 参数")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"Precomputing Protein Features")
        print(f"{'='*70}")
        print(f"Dataset:          {args.dataset}")
        print(f"ESM Model:        {args.esm_model}")
        print(f"Max Seq Length:   {args.max_seq_len}")
        print(f"Finetune:         {args.finetune}")
        print(f"CUDA Device:      {args.device}")
        print(f"Force Recompute:  {args.force}")
        print(f"{'='*70}")
        
        # 执行预计算
        features_path = precompute_protein_features(
            dataset=args.dataset,
            esm_model=args.esm_model,
            max_seq_len=args.max_seq_len,
            finetune=args.finetune,
            device=args.device
        )
        
        print(f"\n{'='*70}")
        print(f"Precomputation Complete!")
        print(f"Features saved to: {features_path}")
        print(f"{'='*70}")