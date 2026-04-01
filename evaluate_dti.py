# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:25:13 2024

@author: Fanding Xu
"""

# ===== 解决 H100 + torch2.1.0 cublas报错 核心代码 (必须放在第一行) =====
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import warnings
warnings.filterwarnings('ignore')
# ======================================================================

import os
import time
import torch
import pickle
import numpy as np
import json
import yaml
import argparse
from utils import set_seed, get_deg_from_list
from trainer.trainer_dti import DTITrainer
from models.model_dti import UnetDTI
from dataset.databuild_dti import get_benchmark_loader


# ===== 添加数据验证函数 =====
def validate_batch(batch):
    """验证批次数据的完整性"""
    g = batch[0]
    
    # 检查节点数量
    num_nodes = g.x.size(0)
    
    # 检查 edge_index 是否越界
    if g.edge_index.numel() > 0:
        max_idx = g.edge_index.max().item()
        if max_idx >= num_nodes:
            print(f"Warning: edge_index max ({max_idx}) >= num_nodes ({num_nodes})")
            return False
    
    # 检查 batch 信息
    if hasattr(g, 'batch'):
        if g.batch.size(0) != num_nodes:
            print(f"Warning: batch size ({g.batch.size(0)}) != num_nodes ({num_nodes})")
            return False
    
    return True


def read_seeds_from_log(dataset_name, split, protein_extractor='cnn'):
    """从日志文件中读取之前运行的seed和对应的run编号"""
    # 根据protein_extractor构建日志文件路径
    if protein_extractor == 'esm':
        log_file = f'log/DTI/{dataset_name}_{split}_esm.txt'
    else:
        log_file = f'log/DTI/{dataset_name}_{split}.txt'
    print(f"Looking for seeds in log file: {log_file}")
    
    if not os.path.exists(log_file):
        print(f"ERROR: Log file {log_file} not found!")
        return None
    
    runs = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'run' in line and '=================' in line:
                # 提取run编号和seed
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        run_idx = int(parts[1])
                        seed = int(parts[-2])
                        runs.append((run_idx, seed))
                    except ValueError:
                        continue
    
    if not runs:
        print(f"ERROR: No runs found in log file {log_file}!")
        return None
    
    print(f"SUCCESS: Found {len(runs)} runs in log file:")
    for run_idx, seed in runs:
        print(f"  Run {run_idx}: seed {seed}")
    
    return runs

def evaluate_single_run(run_idx, seed, model_path, args):
    """评估单个run的DTI模型"""
    try:
        print(f"  Run {run_idx}: Evaluating model {model_path} with seed {seed}...")
        
        # 设置设备
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据 (使用对应的seed)
        loader_tr, loader_va, loader_te = get_benchmark_loader(
            args.dataset, args.split, batch_size=args.batch_size, seed=seed,
            num_workers=args.num_workers, use_esm=(args.protein_extractor == 'esm'),
            use_precomputed_features=args.use_precomputed_features,
            esm_model=args.esm_model
        )
        
        # 计算度分布
        data_list = list({d[0].smiles: d[0] for d in loader_tr.dataset}.values())
        
        # 过滤无效数据
        valid_data_list = []
        for data in data_list:
            if data.edge_index.numel() > 0:
                max_idx = data.edge_index.max().item()
                num_nodes = data.num_nodes
                if max_idx < num_nodes:
                    valid_data_list.append(data)
        
        # 加载配置
        config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)
        config['model']['hidden_dim'] = 128
        
        # 计算度分布
        config['deg'] = get_deg_from_list(valid_data_list)
        
        # 确保度分布在 CPU 上
        if config['deg'].device.type != 'cpu':
            config['deg'] = config['deg'].cpu()
        
        # 设置随机种子
        set_seed(seed)
        
        # 创建模型 - 添加与 benchmark_dti.py 相同的参数
        model = UnetDTI(config, protein_extractor_type=args.protein_extractor,
                       esm_model=args.esm_model, finetune='False',
                       use_precomputed_features=args.use_precomputed_features)
        
        # 加载训练好的权重 (使用对应的模型文件)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            # 检查 checkpoint 结构
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # 新格式：checkpoint 包含 model_state_dict 键
                    # 使用 strict=False 来忽略不匹配的键（模型架构可能略有不同）
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"✅ Loaded model from checkpoint: epoch {checkpoint.get('epoch', 'N/A')}, valid ROC: {checkpoint.get('valid_roc', 'N/A'):.4f}")
                else:
                    # 旧格式：直接是模型状态字典
                    model.load_state_dict(checkpoint, strict=False)
                    print(f"✅ Loaded model from old format checkpoint")
            else:
                # 直接是模型状态字典
                model.load_state_dict(checkpoint, strict=False)
                print(f"✅ Loaded model from direct state dict")
        else:
            print(f"Error: Model path {model_path} not found!")
            return None
        
        # 移动模型到设备
        model = model.to(device)
        
        # 创建训练器
        tr = DTITrainer(args, model, device)
        
        # 运行评估
        results = tr.test(loader_te)
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation for run {run_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_dti(args):
    """评估DTI模型（支持多个run）"""
    try:
        # 创建输出目录
        output_dir = f'log/DTI/evaluation'
        os.makedirs(output_dir, exist_ok=True)
        
        # 确定使用的runs和对应的seed
        # 优先级：seed_file > seeds > log_file
        # 默认从日志文件读取runs和seed，如果没有找到就停止
        if args.seed_file:
            # 从文件中读取seed
            print(f"Using seeds from specified file: {args.seed_file}")
            runs = []
            with open(args.seed_file, 'r') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            seed = int(line)
                            runs.append((idx+1, seed))
                        except ValueError:
                            print(f"Invalid seed in file: {line}")
                            continue
            print(f"Found {len(runs)} seeds in file")
            if not runs:
                print("ERROR: No valid seeds found in specified file!")
                return False
        elif args.seeds:
            # 使用手动指定的seed列表
            seed_list = list(map(int, args.seeds.split(',')))
            runs = [(idx+1, seed) for idx, seed in enumerate(seed_list)]
            print(f"Using manually specified seeds: {seed_list}")
        elif args.use_log_seeds:
            # 必须从日志文件读取runs和seed
            print("[REQUIRED] Must use runs and seeds from log file for proper evaluation...")
            runs = read_seeds_from_log(args.dataset, args.split, args.protein_extractor)
            
            if runs is None:
                # 如果日志文件不存在或没有找到seed，停止运行
                print("ERROR: Failed to get runs and seeds from log file! Evaluation cannot proceed.")
                print("Please ensure the log file exists and contains valid runs and seeds.")
                return False
        else:
            # 单个run模式
            runs = [(1, args.seed)]
            print(f"Using single run with seed: {args.seed}")
        
        # 确定模型路径模式
        if args.model_pattern:
            # 使用 {run} 作为 run index 的占位符
            model_pattern = args.model_pattern.replace('{}', '{run}')
            
            # 使用字符串替换而不是 .format() 来避免占位符冲突
            model_pattern = model_pattern.replace('{dataset}', args.dataset)
            model_pattern = model_pattern.replace('{split}', args.split)
            model_pattern = model_pattern.replace('{protein_extractor}', args.protein_extractor)
            
            print(f"Using model path pattern: {model_pattern}")
        elif args.model_path:
            # 检查是否是单个模型文件
            if os.path.exists(args.model_path):
                # 单个模型文件，所有run都使用同一个模型
                model_pattern = None
                print(f"Using single model: {args.model_path}")
            else:
                # 可能是模式，尝试格式化
                model_pattern = args.model_path.replace('{}', '{run}')
                model_pattern = model_pattern.replace('{dataset}', args.dataset)
                model_pattern = model_pattern.replace('{split}', args.split)
                model_pattern = model_pattern.replace('{protein_extractor}', args.protein_extractor)
                print(f"Using model path as pattern: {model_pattern}")
        else:
            # 默认模型路径模式
            model_pattern = f'checkpoint/DTI/{args.dataset}_{args.split}_{args.protein_extractor}_run{{run}}.pt'
            print(f"Using default model path pattern: {model_pattern}")
        
        all_results = []
        successful_runs = 0
        run_details = []  # 保存每个run的详细信息
        
        print(f"\nStarting evaluation with {len(runs)} runs...")
        print(f"  Dataset: {args.dataset}")
        print(f"  Split: {args.split}")
        print(f"  Protein Extractor: {args.protein_extractor}")
        if args.protein_extractor == 'esm':
            print(f"  ESM Model: {args.esm_model}")
        print(f"  Use Precomputed Features: {args.use_precomputed_features}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Device: {args.device}")
        print("\n" + "="*60)
        
        # 运行多个run
        for run_idx, seed in runs:
            # 获取对应的模型路径
            if model_pattern:
                model_path = model_pattern.format(run=run_idx)
            else:
                model_path = args.model_path
            
            # 评估单个run
            results = evaluate_single_run(run_idx, seed, model_path, args)
            
            if results is not None:
                auroc, auprc, accuracy, sensitivity, specificity, precision = results
                all_results.append(results)
                successful_runs += 1
                
                # 打印单次结果
                print(f"  Run {run_idx} results:")
                print(f"    Model: {model_path}")
                print(f"    Seed: {seed}")
                print(f"    AUROC:    {auroc:.4f}")
                print(f"    AUPRC:    {auprc:.4f}")
                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    Sensitivity: {sensitivity:.4f}")
                print(f"    Specificity: {specificity:.4f}")
                print(f"    Precision: {precision:.4f}")
                
                # 保存单次结果
                run_detail = {
                    'run_idx': run_idx,
                    'seed': seed,
                    'model_path': model_path,
                    'auroc': auroc,
                    'auprc': auprc,
                    'accuracy': accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision
                }
                run_details.append(run_detail)
                
                # Single run results are now saved in the combined results file
            
            print(f"  Run {run_idx}/{len(runs)} completed")
            print("  " + "-"*50)
        
        if successful_runs == 0:
            print("\nAll runs failed! Please check the error messages above.")
            return False
        
        print(f"\nCompleted {successful_runs}/{len(runs)} runs successfully")
        
        # 计算平均指标
        all_results = np.array(all_results)
        mean_results = np.mean(all_results, axis=0)
        std_results = np.std(all_results, axis=0)
        
        # 解析平均结果
        mean_auroc, mean_auprc, mean_accuracy, mean_sensitivity, mean_specificity, mean_precision = mean_results
        std_auroc, std_auprc, std_accuracy, std_sensitivity, std_specificity, std_precision = std_results
        
        # 格式化字符串表示
        formatted_auroc = f"{mean_auroc:.4f} ± {std_auroc:.4f}"
        formatted_auprc = f"{mean_auprc:.4f} ± {std_auprc:.4f}"
        formatted_accuracy = f"{mean_accuracy:.4f} ± {std_accuracy:.4f}"
        formatted_sensitivity = f"{mean_sensitivity:.4f} ± {std_sensitivity:.4f}"
        formatted_specificity = f"{mean_specificity:.4f} ± {std_specificity:.4f}"
        formatted_precision = f"{mean_precision:.4f} ± {std_precision:.4f}"
        
        # 打印平均结果（表格格式）
        print("\n" + "="*100)
        print("Evaluation Results (Mean ± Std)")
        print("="*100)
        print(f"Dataset: {args.dataset} | Split: {args.split} | Runs: {successful_runs}")
        print("="*100)
        print("| {:<12} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} |".format(
            "AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "Precision"
        ))
        print("|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|")
        print("| {:<12} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} |".format(
            formatted_auroc,
            formatted_auprc,
            formatted_accuracy,
            formatted_sensitivity,
            formatted_specificity,
            formatted_precision
        ))
        print("="*100)
        
        # 保存平均结果
        mean_results_dict = {
            # 数值形式
            'mean_auroc': float(mean_auroc),
            'std_auroc': float(std_auroc),
            'mean_auprc': float(mean_auprc),
            'std_auprc': float(std_auprc),
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'mean_sensitivity': float(mean_sensitivity),
            'std_sensitivity': float(std_sensitivity),
            'mean_specificity': float(mean_specificity),
            'std_specificity': float(std_specificity),
            'mean_precision': float(mean_precision),
            'std_precision': float(std_precision),
            
            # 格式化字符串表示
            'formatted_auroc': formatted_auroc,
            'formatted_auprc': formatted_auprc,
            'formatted_accuracy': formatted_accuracy,
            'formatted_sensitivity': formatted_sensitivity,
            'formatted_specificity': formatted_specificity,
            'formatted_precision': formatted_precision,
            
            # 元数据
            'dataset': args.dataset,
            'split': args.split,
            'runs': successful_runs,
            'run_details': run_details,  # 保存每个run的详细信息
            'all_results': all_results.tolist()
        }
        
        if model_pattern:
            mean_results_dict['model_pattern'] = model_pattern
        else:
            mean_results_dict['model_path'] = args.model_path
        
        # 保存平均结果到文件
        if args.protein_extractor == 'esm':
            mean_output_file = os.path.join(output_dir, f'{args.dataset}_{args.split}_esm_mean_results.json')
            text_output_file = os.path.join(output_dir, f'{args.dataset}_{args.split}_esm_results.txt')
        else:
            mean_output_file = os.path.join(output_dir, f'{args.dataset}_{args.split}_mean_results.json')
            text_output_file = os.path.join(output_dir, f'{args.dataset}_{args.split}_results.txt')
        
        with open(mean_output_file, 'w') as f:
            json.dump(mean_results_dict, f, indent=2)
        print(f"\nMean results saved to {mean_output_file}")
        
        # 保存为文本表格格式
        with open(text_output_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("Evaluation Results (Mean ± Std)\n")
            f.write("="*100 + "\n")
            f.write(f"Dataset: {args.dataset} | Split: {args.split} | Protein Extractor: {args.protein_extractor} | Runs: {successful_runs}\n")
            f.write("="*100 + "\n")
            f.write("| {:<12} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} |\n".format(
                "AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "Precision"
            ))
            f.write("|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|\n")
            f.write("| {:<12} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} |\n".format(
                formatted_auroc,
                formatted_auprc,
                formatted_accuracy,
                formatted_sensitivity,
                formatted_specificity,
                formatted_precision
            ))
            f.write("="*100 + "\n")
        print(f"Text table saved to {text_output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DTI model on benchmark datasets')
    
    parser.add_argument('--dataset', type=str, default='bindingdb',
                        help='Benchmark dataset name(default: bindingdb)')
    parser.add_argument('--split', type=str, default='random',
                        help='Split mode: random, cluster, cold, or scaffold(default: random)')
    parser.add_argument('--model_path', type=str,
                        help='Path to the trained model weights (single model)')
    parser.add_argument('--model_pattern', type=str, default='checkpoint/DTI/{dataset}_{split}_{protein_extractor}_run{run}.pt',
                        help='Model path pattern with placeholder for run index (e.g., checkpoint/DTI/bindingdb_esm_run{run}.pt)')
    parser.add_argument('--seed', type=int, default=666779,
                        help='Random seed (default: 666779)')
    parser.add_argument('--seeds', type=str,
                        help='Manual seed list, comma-separated (e.g., 123,456,789)')
    parser.add_argument('--seed_file', type=str,
                        help='Path to file containing seeds (one per line)')
    parser.add_argument('--use_log_seeds', action='store_true',
                        help='Use seeds from previous benchmark runs (read from log file)')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device id (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for data loading (default: 0)')
    parser.add_argument('--protein_extractor', type=str, default='cnn', choices=['cnn', 'esm'],
                        help='Protein feature extractor type (default: cnn)')
    parser.add_argument('--esm_model', type=str, default='esm2_t12_35M_UR50D',
                        help='ESM model name (default: esm2_t12_35M_UR50D)')
    parser.add_argument('--use_precomputed_features', action='store_true',
                        help='Use precomputed protein features instead of extracting during evaluation')
    
    args = parser.parse_args()
    
    # 设置必要的参数
    args.min_epochs = 1
    args.patience = 30
    args.lr = 5e-5
    args.decay = 0
    args.mode = 'cls'
    args.monitor = 'rmse' if args.mode == 'reg' else 'roc'
    
    # 添加DTITrainer需要的额外参数
    args.lr_reduce_rate = 0.5  # 与benchmark_dti.py保持一致
    args.lr_reduce_patience = 50  # 与benchmark_dti.py保持一致
    args.epochs = 100  # 与benchmark_dti.py保持一致
    
    # 加载配置文件
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)
    args.config = json.dumps(config)
    
    print(f"Starting evaluation with parameters:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    if args.model_pattern:
        print(f"  Model pattern: {args.model_pattern}")
    elif args.model_path:
        print(f"  Model path: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Protein Extractor: {args.protein_extractor}")
    if args.protein_extractor == 'esm':
        print(f"  ESM Model: {args.esm_model}")
    print(f"  Use Precomputed Features: {args.use_precomputed_features}")
    print(f"  Use log seeds: {args.use_log_seeds}")
    
    success = evaluate_dti(args)
    
    if success:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed!")