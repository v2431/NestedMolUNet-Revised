# -*- coding: utf-8 -*-
"""
benchmark_dti.py
Created on Wed Jun 19 13:25:13 2024

@author: Fanding Xu, Lizhuo Wang

Example for finding the optimal learning rate:
python benchmark_dti.py \
    --device 6 \
    --find_lr \
    --find_lr_validate 3 \
    --find_lr_smooth both \
    --dataset bindingdb \
    --split cold_protein \
    --protein_extractor esm_cnn \
    --esm_model esm2_t30_150M_UR50D \
    --use_precomputed_features

"""

# ===== 多卡运行时的优化配置 =====
import os
# 只在调试时使用 CUDA_LAUNCH_BLOCKING 和 TORCH_USE_CUDA_DSA
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'  # 6) 长期 benchmark 建议关闭，影响性能/稳定性
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CuBLAS 确定性算法
import torch
# ===== 🔑 可复现性设置 =====
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False  # 关闭 benchmark 以保证可复现
torch.backends.cudnn.deterministic = True  # 启用确定性算法
torch.use_deterministic_algorithms(True)  # 强制所有算子走确定性路径
# 严格可复现：关闭 TF32（避免不同 GPU/驱动版本产生数值差异）
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import warnings
warnings.filterwarnings('ignore')
# ======================================================================

# ======================================================================

import sys
import time
import pickle
import numpy as np
import json
import yaml
import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from utils import set_seed, get_deg_from_list
from trainer.trainer_dti import DTITrainer
from models.model_dti import UnetDTI
from dataset.databuild_dti import get_benchmark_loader


# ===== 🔍 第一步:精准诊断工具(移到全局) =====
class NaNDetector:
    """最小侵入式的NaN探测器"""
    
    # 关键模块名称模式（只 hook 这些）
    KEYWORD_PATTERNS = ['bcn', 'mlp_classifier', 'unet', 'esm', 'protein_extractor', 'proj', 'conv']
    
    def __init__(self, model, save_on_nan=True):
        self.model = model
        self.save_on_nan = save_on_nan
        self.nan_found = False
        self.culprit_module = None
        self.hooks = []
        
    @staticmethod
    def _safe_tensor_info(t):
        """安全地提取 tensor 统计信息（不保存整个 tensor）"""
        if not isinstance(t, torch.Tensor):
            return str(type(t))
        return {
            'shape': tuple(t.shape),
            'dtype': str(t.dtype),
            'device': str(t.device),
            'nan_count': int(torch.isnan(t).sum().item()) if torch.is_floating_point(t) else 0,
            'min': float(t.min().detach().cpu()) if torch.is_floating_point(t) else None,
            'max': float(t.max().detach().cpu()) if torch.is_floating_point(t) else None,
            'mean': float(t.mean().detach().cpu()) if torch.is_floating_point(t) else None,
        }
    
    def _should_monitor(self, name):
        """判断是否应该监控该模块"""
        name_lower = name.lower()
        return any(kw in name_lower for kw in self.KEYWORD_PATTERNS)
        
    def register_hooks(self):
        """只注册监控hook,不做任何修改"""
        def make_hook(name):
            def hook(module, args, kwargs, output):
                # with_kwargs=True 时，签名是 (module, args, kwargs, output)
                # args 是 tuple，kwargs 是 dict
                if self.nan_found:  # 已经找到第一个NaN,跳过后续
                    return
                
                # 检查输入 (args)
                for i, inp in enumerate(args if isinstance(args, tuple) else [args]):
                    if isinstance(inp, torch.Tensor) and torch.isnan(inp).any():
                        print(f"❌ NaN in INPUT[{i}] of {name}")
                        print(f"   Info: {self._safe_tensor_info(inp)}")
                        self.nan_found = True
                        self.culprit_module = name
                        if self.save_on_nan:
                            # 只保存轻量信息，不保存完整 state_dict
                            torch.save({
                                'module_name': name,
                                'input_info': self._safe_tensor_info(inp),
                                'timestamp': time.time()
                            }, f'nan_culprit_{name.replace(".", "_")}.pt')
                        return
                
                # 检查输出
                outputs = output if isinstance(output, tuple) else [output]
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                        print(f"🔥 NaN FIRST APPEARED in OUTPUT[{i}] of {name}")
                        print(f"   Info: {self._safe_tensor_info(out)}")
                        # 安全地打印输入统计
                        if isinstance(args, tuple) and len(args) > 0:
                            inp0 = args[0]
                            if isinstance(inp0, torch.Tensor):
                                print(f"   Input[0] info: {self._safe_tensor_info(inp0)}")
                        self.nan_found = True
                        self.culprit_module = name
                        if self.save_on_nan:
                            # 只保存轻量信息，不保存完整 state_dict
                            torch.save({
                                'module_name': name,
                                'output_info': self._safe_tensor_info(out),
                                'timestamp': time.time()
                            }, f'nan_culprit_{name.replace(".", "_")}.pt')
                        return
            return hook
        
        for name, module in self.model.named_modules():
            # 只监控关键模块（而不是所有叶子节点）
            if self._should_monitor(name):
                handle = module.register_forward_hook(make_hook(name), with_kwargs=True)
                self.hooks.append(handle)
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ===== 数据验证函数 =====
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


def run_single_experiment(run_id, args, seed, batch_size, device_id, return_dict):
    """在独立进程中运行单次实验"""
    import copy
    args = copy.deepcopy(args)  # 避免修改父进程的 args
    args.run_id = run_id
    args.device_id = device_id
    # 为每个进程创建独立的日志文件
    run_log_file = f'log/DTI/detail/{args.dataset}_{args.split}_{args.protein_extractor}_run{run_id}_gpu{device_id}.log'
    os.makedirs(os.path.dirname(run_log_file), exist_ok=True)
    
    # ===== 保存原始输出 =====
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # ===== 创建一个自定义的输出类 =====
    class DualOutput:
        """普通日志写文件；tqdm进度条(含\\r)只写终端，覆盖显示不刷屏"""
        def __init__(self, file_path, original_stream):
            self.file = open(file_path, 'w', buffering=1)  # 行缓冲
            self.terminal = original_stream
            
        def write(self, message):
            # tqdm 进度条的核心特征：使用 \r 回车覆盖同一行
            if '\r' in message:
                self.terminal.write(message)
                self.terminal.flush()
                return
            
            # 写入日志文件时，去除开头的换行符（避免多余空行）
            log_message = message.lstrip('\n')
            if log_message:
                self.file.write(log_message)
                self.file.flush()
            
        def flush(self):
            self.file.flush()
            self.terminal.flush()
            
        def isatty(self):
            # 让 tqdm 更倾向于“动态覆盖行”的行为
            return True
            
        def close(self):
            self.file.close()
    
    # ===== 初始化 dual_output =====
    dual_output = None
    
    # ===== 定义辅助函数 =====
    def log_info(message):
        """只输出到日志文件"""
        if dual_output:
            dual_output.file.write(f"{message}\n")
            dual_output.file.flush()
    
    def print_info(message):
        """只输出到终端（并避免和tqdm进度条混行）"""
        # 先回到行首（清除可能的进度条残留），再打印消息
        original_stdout.write(f"[Run {run_id}] {message}\n")
        original_stdout.flush()
    
    def log_and_print(message):
        """同时输出到日志文件和终端"""
        log_info(message)
        print_info(message)
    
    # 本地结果存储，避免过早访问return_dict
    local_result = None
    
    try:
        # ===== 创建 DualOutput 并重定向 stdout =====
        dual_output = DualOutput(run_log_file, original_stdout)
        sys.stdout = dual_output
        sys.stderr = dual_output
        
        # ===== 🔑 CRITICAL: GPU 存在性检查 =====
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if device_id >= torch.cuda.device_count():
            raise RuntimeError(f"device_id={device_id} out of range, cuda.device_count={torch.cuda.device_count()}")
        
        # 设置设备
        device = torch.device(f'cuda:{device_id}')
        
        # ===== 关键信息同时输出到终端 =====
        log_and_print(f"[GPU {device_id}][Seed {seed}] Starting experiment...")
        log_and_print(f"Dataset: {args.dataset}, Split: {args.split}")
        if args.protein_extractor in ['esm', 'esm_cnn']:
            log_and_print(f"Protein: {args.protein_extractor}, ESM: {args.esm_model}")
        else:
            log_and_print(f"Protein: {args.protein_extractor}")
        log_and_print(f"Batch: {batch_size}, Epochs: {args.epochs}, Patience: {args.patience}")
        
        # 加载数据
        log_and_print("Loading data...")
        loader_tr, loader_va, loader_te = get_benchmark_loader(
            args.dataset, args.split, batch_size=batch_size, seed=seed,
            num_workers=args.num_workers, use_esm=(args.protein_extractor in ['esm', 'esm_cnn']),
            use_precomputed_features=args.use_precomputed_features,
            esm_model=args.esm_model,
            max_seq_len=args.max_seq_len,
            features_suffix=args.features_suffix
        )
        log_and_print(f"Dataset: {len(loader_tr.dataset)} | {len(loader_va.dataset)} | {len(loader_te.dataset)}")
        
        # 数据验证
        log_and_print("Validating data...")
        valid_count = 0
        for idx, batch in enumerate(loader_tr):
            if not validate_batch(batch):
                log_and_print(f"Invalid batch at {idx}")
                return_dict[run_id] = {'success': False, 'error': 'Invalid batch'}
                return
            valid_count += 1
            if idx >= 10:
                break
        log_and_print(f"Validation complete: {valid_count} batches OK")
        
        # 验证第一个批次的蛋白质特征形状
        for idx, batch in enumerate(loader_tr):
            # 3) 统一用 batch[:3] 解包，避免字段数量变化时崩溃
            g, tars, ys = batch[:3]
            meta = batch[3:]  # 可能存在的额外字段
            if isinstance(tars, dict):
                log_and_print(f"First batch - tars keys: {list(tars.keys())}")
                if 'input_ids' in tars:
                    log_and_print(f"  input_ids: {tars['input_ids'].shape}, dtype: {tars['input_ids'].dtype}")
                if 'tokens' in tars:
                    log_and_print(f"  tokens: {tars['tokens'].shape}, dtype: {tars['tokens'].dtype}")
                if 'features' in tars:
                    log_and_print(f"  features: {tars['features'].shape}, dtype: {tars['features'].dtype}")
                if 'attention_mask' in tars:
                    m = tars['attention_mask']
                    log_and_print(f"  attention_mask: {m.shape}, dtype: {m.dtype}")
                    log_and_print(f"  mask valid_len: min={int(m.sum(1).min())}, max={int(m.sum(1).max())}, mean={float(m.sum(1).float().mean()):.1f}")
                    log_and_print(f"  mask unique values: {torch.unique(m).cpu().tolist()}")
            else:
                log_and_print(f"First batch - tars shape: {tars.shape}, dtype: {tars.dtype}")
            log_and_print(f"First batch - g.x shape: {g.x.shape}, dtype: {g.x.dtype}")
            log_and_print(f"First batch - ys shape: {ys.shape}, dtype: {ys.dtype}")
            break
        
        # 清空显存
        torch.cuda.empty_cache()
        
        # 计算度分布
        data_list = list({d[0].smiles: d[0] for d in loader_tr.dataset}.values())
        valid_data_list = [d for d in data_list
                          if d.edge_index.numel() > 0 and
                          d.edge_index.max().item() < d.num_nodes]
        
        log_and_print(f"Valid molecules: {len(valid_data_list)}/{len(data_list)}")
        
        # 加载配置
        config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)
        config['model']['hidden_dim'] = 128
        config['deg'] = get_deg_from_list(valid_data_list)
        
        if config['deg'].device.type != 'cpu':
            config['deg'] = config['deg'].cpu()
        
        # 设置随机种子
        set_seed(seed)
        
        # ===== 🔑 预计算特征兼容性检查 =====
        if args.use_precomputed_features:
            if args.protein_extractor not in ['esm', 'esm_cnn']:
                raise ValueError(f"Precomputed features only supported for protein_extractor='esm' or 'esm_cnn', got {args.protein_extractor}")
            log_and_print("⚠️  Using precomputed features - expecting ESM backbone will NOT be loaded (will verify below)")
        
        # 创建模型
        model = UnetDTI(config, protein_extractor_type=args.protein_extractor,
                       esm_model=args.esm_model, finetune='False',
                       use_precomputed_features=args.use_precomputed_features)
        
        # ===== 🔑 检查ESM模块是否真的没加载 =====
        total_params = sum(p.numel() for p in model.parameters())
        log_and_print(f"Total model parameters: {total_params:,}")
        
        # 检查是否有ESM模块（排除明显的小层如esm_proj/esm_adapter）
        excluded_keywords = ['proj', 'adapter', 'feature', 'head']  # 排除这些关键词的层
        has_esm_backbone = False
        for name, _ in model.named_modules():
            name_lower = name.lower()
            if 'esm' in name_lower and not any(kw in name_lower for kw in excluded_keywords):
                has_esm_backbone = True
                break
        log_and_print(f"ESM backbone present (excluding small layers): {has_esm_backbone}")
        
        # 用参数量作为辅助判断
        if args.use_precomputed_features and total_params > 40_000_000:
            log_and_print("⚠️  Notice: Model has >40M params - double-check if ESM backbone is loaded as expected")
        
        # 🔧 注册NaN探测器
        detector = NaNDetector(model, save_on_nan=True)
        detector.register_hooks()
        log_and_print("NaN detector registered")
        
        # 打印模型信息
        log_and_print(f"Model initialized: {model.__class__.__name__}")
        log_and_print(f"Protein extractor: {args.protein_extractor}")
        if args.protein_extractor in ['esm', 'esm_cnn']:
            log_and_print(f"ESM model: {args.esm_model}")
        log_and_print(f"Use precomputed features: {args.use_precomputed_features}")
        
        # 🔧 创建Trainer并配置梯度裁剪
        log_and_print("Starting training...")
        tr = DTITrainer(args, model, device)
        
        # 在Trainer内部启用梯度裁剪(需要修改trainer代码,见下方)
        tr.use_grad_clip = True
        tr.max_grad_norm = 1.0
        
        # 关闭混合精度
        tr.use_amp = False
        
        # ===== 训练过程 =====
        # 注意：tr() 内部已经在训练结束后加载 best checkpoint 并测试
        # 所以 train_results 就是最终的 test results
        train_results = None  # 🔧 初始化为 None
        
        # 🔧 新增：如果 args.test_only 为 True，临时设置 epochs=0 跳过训练循环
        if getattr(args, 'test_only', False):
            base_save_path = f'checkpoint/DTI/{args.dataset}_{args.split}_{args.protein_extractor}_lr{args.lr}_bs{args.batch_size}_run{run_id}'
            best_ckpt_path = f'{base_save_path}_global_best_roc.pt'
            log_and_print(f"\n🔍 Test-only mode: loading checkpoint from {best_ckpt_path}")
            
            # 🔧 临时将 epochs 设为 0，跳过训练循环但保留测试逻辑
            original_epochs = args.epochs
            args.epochs = 0
            
            try:
                train_results = tr(loader_tr, loader_va, loader_te, tensorboard=False,
                              save_path=base_save_path, load_path=best_ckpt_path)
            except Exception as e:
                log_and_print(f"Testing exception: {e}")
                import traceback
                log_and_print(traceback.format_exc())
                local_result = {'success': False, 'error': str(e)}
                args.epochs = original_epochs
                return
            finally:
                args.epochs = original_epochs
        else:
        # 🔧 原始的训练代码
            try:
                # 基础路径（不含后缀）
                base_save_path = f'checkpoint/DTI/{args.dataset}_{args.split}_{args.protein_extractor}_lr{args.lr}_bs{args.batch_size}_run{run_id}'
                train_results = tr(loader_tr, loader_va, loader_te, tensorboard=False,
                              save_path=base_save_path)
            except Exception as e:
                log_and_print(f"Training exception: {e}")
                import traceback
                log_and_print(traceback.format_exc())
                
                # 🔧 即使训练失败，也返回部分结果
                local_result = {
                    'success': False,
                    'error': str(e),
                    'train_results': None
                }
                detector.remove_hooks()
                return
        
        # 🔧 移除hooks并检查结果
        detector.remove_hooks()
        if detector.nan_found:
            log_and_print(f"\n🎯 罪魁祸首: {detector.culprit_module}")
            log_and_print(f"NaN现场已保存到 nan_culprit_{detector.culprit_module.replace('.', '_')}.pt")
            local_result = {
                'success': False,
                'error': f'NaN detected in {detector.culprit_module}'
            }
            return
        
        # 🔧 检查 train_results 是否有效
        if train_results is None:
            log_and_print("Training returned None")
            local_result = {
                'success': False,
                'error': 'Training returned None'
            }
            return
        
        # ===== 使用训练结束时的 test results =====
        # tr() 内部已经加载 best checkpoint 并测试，无需再次测试
        # 新的返回格式是结构化字典，包含 global_roc/global_ef 两种模型的两种阈值结果
        if isinstance(train_results, dict) and 'global_roc' in train_results:
            # 新的结构化格式
            best_epoch_global = train_results.get('best_epoch_global_roc', 0)
            best_epoch_perprot = train_results.get('best_epoch_global_ef', 0)
            
            # 提取 global 模型的 F1 阈值结果作为默认 metric（兼容旧逻辑）
            global_f1_results = train_results['global_roc']['f1']
            if global_f1_results is not None and len(global_f1_results) >= 8:
                test_results = global_f1_results[:8]  # [auroc, auprc, positive_rate, accuracy, sensitivity, specificity, precision, threshold]
                metric = test_results[0]  # AUROC
            else:
                test_results = None
                metric = 0.0
            
            # 保存完整结构化结果用于后续汇总
            full_results = train_results
        else:
            # 兼容旧格式
            test_results = train_results
            best_epoch_global = 0
            best_epoch_perprot = 0
            full_results = None
            if args.monitor == 'roc' and test_results is not None:
                metric = test_results[0]
            elif test_results is not None:
                metric = test_results[0] if isinstance(test_results, (list, tuple)) else 0.0
            else:
                metric = 0.0
        
        # ===== 🔑 CRITICAL: 训练结果格式校验 =====
        if test_results is not None:
            if not (isinstance(test_results, (list, tuple)) and len(test_results) == 8):
                raise ValueError(f"Unexpected test_results format: {type(test_results)} len={len(test_results) if hasattr(test_results,'__len__') else 'NA'}")
        
        # 4) 只回传轻量汇总，避免 mp.Manager().dict() 传大对象拖慢/卡住
        # 详细结果写文件，主进程再汇总
        local_result = {
            'success': True,
            'run_id': run_id,  # 2) 添加 run_id，确保续跑/并行时能对上 detail 文件
            'metric': metric,
            'results': test_results,  # 8值列表 [auroc, auprc, ...]
            'seed': seed,
            'best_epoch_global': best_epoch_global,
            'best_epoch_perprot': best_epoch_perprot,
            # 只提取关键阈值信息，不传完整 full_results
            'threshold_f1': full_results.get('global_roc', {}).get('threshold_f1', 0.5) if isinstance(full_results, dict) else 0.5,
            'threshold_youden': full_results.get('global_roc', {}).get('threshold_youden', 0.5) if isinstance(full_results, dict) else 0.5,
            'valid_positive_rate': full_results.get('global_roc', {}).get('valid_positive_rate', 0.0) if isinstance(full_results, dict) else 0.5,
        }
        
        # 将详细结构化结果写入独立文件（每个run一个文件）
        # 文件名包含 lr 信息，避免不同 lr 的结果互相覆盖
        if isinstance(full_results, dict):
            try:
                detail_dir = f'log/DTI/detail/{args.dataset}_{args.split}_{args.protein_extractor}'
                os.makedirs(detail_dir, exist_ok=True)
                # 格式化 lr 为字符串（如 3e-05 -> 3e-05）
                lr_str = f"{args.lr:.1e}".replace('e-0', 'e-').replace('e+0', 'e+')
                detail_file = f'{detail_dir}/lr{lr_str}_run{run_id}_seed{seed}_detail.json'
                with open(detail_file, 'w') as f:
                    json.dump(full_results, f, indent=2)
                log_and_print(f"Detailed results saved to: {detail_file}")
            except Exception as e:
                log_and_print(f"Warning: Failed to save detailed results: {e}")
        
        log_and_print(f"Completed: {args.monitor}={metric:.4f}")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        log_info(error_msg)
        print_info(error_msg)
        
        import traceback
        log_info(traceback.format_exc())
        
        local_result = {
            'success': False,
            'error': str(e)
        }
    finally:
        # 清理显存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ===== 恢复 stdout 并关闭文件 =====
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        if dual_output is not None:
            dual_output.close()
        
        # 尝试将结果写入return_dict，处理可能的连接问题
        try:
            if local_result is not None:
                return_dict[run_id] = local_result
                
                # 输出摘要
                if local_result.get('success', False):
                    print(f"[Summary] Run {run_id} (GPU {device_id}) completed successfully")
                else:
                    print(f"[Summary] Run {run_id} (GPU {device_id}) failed: {local_result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"[Warning] Failed to write result to return_dict: {str(e)}")


# ===== 格式化评估结果的函数 =====
def format_evaluation_results(test_results):
    """格式化评估结果为表格形式"""
    metrics = {
        'roc': [],
        'prc': [],
        'positive_rate': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'threshold': []
    }
    
    for result in test_results:
        if result is None:
            continue
        
        # 处理字典格式（新格式：包含 success, results, metric 等字段）
        if isinstance(result, dict):
            if result.get('success') and 'results' in result:
                r = result['results']
                # 处理新的 8/9 值格式
                if isinstance(r, (list, tuple)) and len(r) >= 8:
                    metrics['roc'].append(r[0])
                    metrics['prc'].append(r[1])
                    metrics['positive_rate'].append(r[2])
                    metrics['accuracy'].append(r[3])
                    metrics['sensitivity'].append(r[4])
                    metrics['specificity'].append(r[5])
                    metrics['precision'].append(r[6])
                    metrics['threshold'].append(r[7])
        # 处理直接的 list/tuple 格式（旧格式）
        elif isinstance(result, (list, tuple)) and len(result) >= 8:
            metrics['roc'].append(result[0])
            metrics['prc'].append(result[1])
            metrics['positive_rate'].append(result[2])
            metrics['accuracy'].append(result[3])
            metrics['sensitivity'].append(result[4])
            metrics['specificity'].append(result[5])
            metrics['precision'].append(result[6])
            metrics['threshold'].append(result[7])
        # 兼容旧格式 6 值
        elif isinstance(result, (list, tuple)) and len(result) == 6:
            metrics['roc'].append(result[0])
            metrics['prc'].append(result[1])
            metrics['accuracy'].append(result[2])
            metrics['sensitivity'].append(result[3])
            metrics['specificity'].append(result[4])
            metrics['precision'].append(result[5])
    
    metrics_dict = {}
    for metric_name, values in metrics.items():
        if len(values) > 0:
            values_np = np.array(values)
            metrics_dict[metric_name] = {
                'mean': values_np.mean(),
                'std': values_np.std()
            }
    
    # 主对比（阈值无关）
    primary_metrics = ['roc', 'prc', 'positive_rate']
    primary_names = ['AUROC', 'AUPRC', 'AUPRC Baseline']
    
    # 辅助对比（阈值相关）
    secondary_metrics = ['accuracy', 'sensitivity', 'specificity', 'precision']
    secondary_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision']
    
    # 构建主对比表格
    header1 = "| " + " | ".join([f"{name:<15}" for name in primary_names]) + " |"
    separator1 = "|" + "|".join(["-" * 17 for _ in primary_names]) + "|"
    
    values1 = []
    for metric in primary_metrics:
        if metric in metrics_dict:
            values1.append(f"{metrics_dict[metric]['mean']:.4f} ± {metrics_dict[metric]['std']:.4f}")
        else:
            values1.append("N/A")
    
    data_row1 = "| " + " | ".join([f"{val:<15}" for val in values1]) + " |"
    
    # 构建辅助对比表格
    header2 = "| " + " | ".join([f"{name:<12}" for name in secondary_names]) + " |"
    separator2 = "|" + "|".join(["-" * 14 for _ in secondary_names]) + "|"
    
    values2 = []
    for metric in secondary_metrics:
        if metric in metrics_dict:
            values2.append(f"{metrics_dict[metric]['mean']:.4f} ± {metrics_dict[metric]['std']:.4f}")
        else:
            values2.append("N/A")
    
    data_row2 = "| " + " | ".join([f"{val:<12}" for val in values2]) + " |"
    
    formatted_table = f"Primary Metrics (Threshold-independent):\n{header1}\n{separator1}\n{data_row1}\n\nSecondary Metrics (Threshold-dependent):\n{header2}\n{separator2}\n{data_row2}"
    
    return formatted_table, metrics_dict


def save_evaluation_results(args, valid_test_results):
    """保存评估结果到文件"""
    eval_dir = 'log/DTI/evaluation'
    os.makedirs(eval_dir, exist_ok=True)
    
    formatted_table, metrics_dict = format_evaluation_results(valid_test_results)
    
    txt_file = f'{eval_dir}/{args.dataset}_{args.split}_{args.protein_extractor}_results.txt'
    with open(txt_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Protein Extractor: {args.protein_extractor}\n")
        if args.protein_extractor in ['esm', 'esm_cnn']:
            f.write(f"ESM Model: {args.esm_model}\n")
        f.write(f"Runs: {len(valid_test_results)}\n\n")
        f.write(formatted_table)
    
    json_file = f'{eval_dir}/{args.dataset}_{args.split}_{args.protein_extractor}_mean_results.json'
    json_data = {
        'dataset': args.dataset,
        'split': args.split,
        'protein_extractor': args.protein_extractor,
        'esm_model': args.esm_model if args.protein_extractor == 'esm' else None,
        'runs': len(valid_test_results),
        'metrics': metrics_dict
    }
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return txt_file, json_file


def save_structured_results(args, all_results):
    """
    保存结构化结果到 JSON/CSV，方便模型间对比
    
    包含每个 run 的完整信息：
    - split, seed, model
    - AUROC, AUPRC, AUPRC_baseline
    - t_f1, t_youden
    - Acc/Sens/Spec/Prec @ 两阈值
    """
    eval_dir = 'log/DTI/evaluation'
    os.makedirs(eval_dir, exist_ok=True)
    
    structured_data = []
    
    for result in all_results:
        if result is None:
            continue
        
        # 2.1) 直接从 result 读取 seed，避免外部 seeds 错位
        seed = result.get('seed', 0)
        rid = result.get('run_id')
        if rid is None:
            continue
        
        # 5) 优先从 detail JSON 文件读取完整结果，否则用轻量结果
        full = None
        # ✅ Fix 2：使用 glob 匹配（与写入路径一致），兼容 lr 前缀
        detail_dir = f'log/DTI/detail/{args.dataset}_{args.split}_{args.protein_extractor}'
        detail_pattern = f'{detail_dir}/*_run{rid}_seed{seed}_detail.json'
        detail_files = list(Path('.').glob(detail_pattern))
        
        if detail_files:
            detail_file = detail_files[0]  # 取第一个匹配（通常只有一个）
            try:
                with open(detail_file, 'r') as f:
                    full = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load detail file {detail_file}: {e}")
        
        # 如果文件读取失败，尝试用 result 中的轻量信息（兼容旧逻辑）
        if full is None:
            # 从 result['results'] (8值列表) 提取信息
            test_res = result.get('results')
            if test_res is not None and len(test_res) == 8:
                auroc, auprc, test_pos_rate, acc, sens, spec, prec, _ = test_res
                structured_data.append({
                    'split': args.split,
                    'seed': seed,
                    'model': args.protein_extractor,
                    'model_type': 'global',
                    'threshold_type': 'f1',
                    'AUROC': auroc,
                    'AUPRC': auprc,
                    'AUPRC_baseline': test_pos_rate,
                    'threshold': result.get('threshold_f1', 0.5),
                    'Accuracy': acc,
                    'Sensitivity': sens,
                    'Specificity': spec,
                    'Precision': prec,
                    'best_epoch': result.get('best_epoch_global', 0),
                    'valid_positive_rate': result.get('valid_positive_rate', 0.0),
                })
            continue
        
        # 从完整结果中提取两种阈值的信息
        global_results = full.get('global', {})
        perprot_results = full.get('perprot', {})
        
        # Global model F1 阈值结果
        global_f1 = global_results.get('f1')
        if global_f1 is not None and len(global_f1) == 8:
            auroc, auprc, test_pos_rate, acc, sens, spec, prec, _ = global_f1
            structured_data.append({
                'split': args.split,
                'seed': seed,
                'model': args.protein_extractor,
                'model_type': 'global',
                'threshold_type': 'f1',
                'AUROC': auroc,
                'AUPRC': auprc,
                'AUPRC_baseline': test_pos_rate,
                'threshold': global_results.get('threshold_f1', 0.5),
                'Accuracy': acc,
                'Sensitivity': sens,
                'Specificity': spec,
                'Precision': prec,
                'best_epoch': full.get('best_epoch_global', 0),
                'valid_positive_rate': global_results.get('valid_positive_rate', 0.0),
            })
        
        # Global model Youden 阈值结果
        global_youden = global_results.get('youden')
        if global_youden is not None and len(global_youden) == 8:
            auroc, auprc, test_pos_rate, acc, sens, spec, prec, _ = global_youden
            structured_data.append({
                'split': args.split,
                'seed': seed,
                'model': args.protein_extractor,
                'model_type': 'global',
                'threshold_type': 'youden',
                'AUROC': auroc,
                'AUPRC': auprc,
                'AUPRC_baseline': test_pos_rate,
                'threshold': global_results.get('threshold_youden', 0.5),
                'Accuracy': acc,
                'Sensitivity': sens,
                'Specificity': spec,
                'Precision': prec,
                'best_epoch': full.get('best_epoch_global', 0),
                'valid_positive_rate': global_results.get('valid_positive_rate', 0.0),
            })
        
        # Per-protein model F1 阈值结果
        perprot_f1 = perprot_results.get('f1')
        if perprot_f1 is not None and len(perprot_f1) == 8:
            auroc, auprc, test_pos_rate, acc, sens, spec, prec, _ = perprot_f1
            structured_data.append({
                'split': args.split,
                'seed': seed,
                'model': args.protein_extractor,
                'model_type': 'perprot',
                'threshold_type': 'f1',
                'AUROC': auroc,
                'AUPRC': auprc,
                'AUPRC_baseline': test_pos_rate,
                'threshold': perprot_results.get('threshold_f1', 0.5),
                'Accuracy': acc,
                'Sensitivity': sens,
                'Specificity': spec,
                'Precision': prec,
                'best_epoch': full.get('best_epoch_perprot', 0),
                'valid_positive_rate': perprot_results.get('valid_positive_rate', 0.0),
            })
        
        # Per-protein model Youden 阈值结果
        perprot_youden = perprot_results.get('youden')
        if perprot_youden is not None and len(perprot_youden) == 8:
            auroc, auprc, test_pos_rate, acc, sens, spec, prec, _ = perprot_youden
            structured_data.append({
                'split': args.split,
                'seed': seed,
                'model': args.protein_extractor,
                'model_type': 'perprot',
                'threshold_type': 'youden',
                'AUROC': auroc,
                'AUPRC': auprc,
                'AUPRC_baseline': test_pos_rate,
                'threshold': perprot_results.get('threshold_youden', 0.5),
                'Accuracy': acc,
                'Sensitivity': sens,
                'Specificity': spec,
                'Precision': prec,
                'best_epoch': full.get('best_epoch_perprot', 0),
                'valid_positive_rate': perprot_results.get('valid_positive_rate', 0.0),
            })
    
    # 保存为 JSON
    json_file = f'{eval_dir}/{args.dataset}_{args.split}_{args.protein_extractor}_structured.json'
    with open(json_file, 'w') as f:
        json.dump(structured_data, f, indent=2)
    
    # 保存为 CSV
    csv_file = f'{eval_dir}/{args.dataset}_{args.split}_{args.protein_extractor}_structured.csv'
    if structured_data:
        import csv
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=structured_data[0].keys())
            writer.writeheader()
            writer.writerows(structured_data)
    
    return json_file, csv_file


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='bindingdb',
                        help='Bnechmark dataset name(default: bindingdb)')
    parser.add_argument('--split', type=str, default='random',
                        help='Split mode: random, scaffold, cold_protein, cold_compound, cold_pair, blind_start (default: random)')
    parser.add_argument('--runs', type=int, default=5,
                        help='indepent run times (default: 5)')
    parser.add_argument('--run_start', type=int, default=1, help='Start run index (for resuming runs)')
    parser.add_argument('--run_end', type=int, default=-1, help='End run index (for resuming runs, -1 means run all from start)')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting completed runs')
    parser.add_argument('--test_only', action='store_true', help='Only test existing checkpoints, no training')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='learning rate reduce rate (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', default=15, type=int,
                        help='learning rate reduce patience (default: 15)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='maximum training epochs (default: 100)')
    parser.add_argument('--log_train_results', action="store_false",
                        help='whether to evaluate training set in each epoch, costs more time (default: True)')
    parser.add_argument('--device', type=int, default=1,
                        help='CUDA device id (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for data loading (default: 4)')
    parser.add_argument('--patience', type=int, default=30,
                        help='patience for early stopping (default: 30)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--protein_extractor', type=str, default='cnn', choices=['cnn', 'esm', 'esm_cnn'],
                        help='Protein feature extractor type (default: cnn)')
    parser.add_argument('--esm_model', type=str, default='esm2_t12_35M_UR50D',
                        help='ESM model name (default: esm2_t12_35M_UR50D)')
    parser.add_argument('--use_precomputed_features', action='store_true',
                        help='Use precomputed protein features instead of extracting during training')
    parser.add_argument('--max_seq_len', type=int, default=1200,
                        help='Maximum protein sequence length (default: 1200)')
    parser.add_argument('--features_suffix', type=str, default='',
                        help='Suffix for precomputed features file (default: "")')
    parser.add_argument('--parallel', action='store_true',
                        help='Run multiple experiments in parallel using multiple GPUs')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Comma-separated list of GPU IDs to use (default: 0,1,2,3)')
    parser.add_argument('--find_lr', action='store_true',
                        help='Run LR Finder to find optimal learning rate (no training)')
    parser.add_argument('--find_lr_iter', type=int, default=None,
                        help='Number of iterations for LR Finder (default: auto, 200-500)')
    parser.add_argument('--find_lr_smooth', type=str, default='both',
                        choices=['ema', 'savgol', 'both'],
                        help='Smoothing method for LR Finder (default: both)')
    parser.add_argument('--find_lr_validate', type=int, default=3,
                        help='Number of validation epochs for LR Finder (default: 3, 0 to disable)')
    parser.add_argument('--find_lr_auto_range', action='store_true', default=True,
                        help='Auto-adjust LR search range based on model size (default: True)')
    args = parser.parse_args()
    
    # ===== LR Finder 模式 =====
    if getattr(args, 'find_lr', False):
        print(f"\n{'='*70}")
        print(f"LR Finder Mode")
        print(f"{'='*70}")
        
        # 设置设备
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 加载配置
        config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)
        config['model']['hidden_dim'] = 128                    # ✅ 与正式训练一致
        args.config = json.dumps(config)
        args.mode = 'cls'
        args.monitor = 'roc'
        args.min_epochs = 30                                   # ✅ DTITrainer.__init__ 需要此字段
        
        # ✅ Fix 1：与正式训练保持一致的 loader 调用
        seed = 42  # LR Finder 用固定种子，结果可复现
        loader_tr, loader_va, loader_te = get_benchmark_loader(
            args.dataset, args.split, 
            batch_size=args.batch_size, 
            seed=seed, 
            num_workers=args.num_workers, 
            use_esm=(args.protein_extractor in ['esm', 'esm_cnn']),
            use_precomputed_features=args.use_precomputed_features, 
            esm_model=args.esm_model, 
            max_seq_len=args.max_seq_len, 
            features_suffix=args.features_suffix
        )
        print(f"Dataset loaded: {len(loader_tr.dataset)} train samples")
        
        # ✅ Fix A：计算 deg（与正式训练一致）
        data_list = list({d[0].smiles: d[0] for d in loader_tr.dataset}.values())
        valid_data_list = [d for d in data_list 
                           if d.edge_index.numel() > 0 and 
                           d.edge_index.max().item() < d.num_nodes]
        config['deg'] = get_deg_from_list(valid_data_list)
        if config['deg'].device.type != 'cpu':
            config['deg'] = config['deg'].cpu()
        
        # ✅ Fix 2：与正式训练完全一致的模型构造
        set_seed(seed)
        model = UnetDTI(config, 
                        protein_extractor_type=args.protein_extractor, 
                        esm_model=args.esm_model, 
                        finetune='False', 
                        use_precomputed_features=args.use_precomputed_features)
        # ✅ Fix 1：删除冗余的 model.to(device)，DTITrainer.__init__ 内部统一管理
        
        # ✅ Fix 3 & 4：直接使用 DTITrainer（内部已创建 AdamW），不再手动替换 optimizer
        # DTITrainer.__init__ 会创建 AdamW，与正式训练完全一致
        tr = DTITrainer(args, model, device)
        # tr.optimizer 已经是正确的 AdamW，无需覆盖
        
        # 运行 LR Finder
        os.makedirs('log/DTI', exist_ok=True)
        save_plot_path = f'log/DTI/lr_finder_{args.dataset}_{args.split}_{args.protein_extractor}.png'
        
        result = tr.find_lr(
            loader_tr, 
            start_lr=1e-7, 
            end_lr=1, 
            num_iter=args.find_lr_iter,
            save_plot_path=save_plot_path,
            smooth_method=args.find_lr_smooth,
            num_validation_epochs=args.find_lr_validate,
            auto_range=args.find_lr_auto_range,
        )
        
        # 处理返回结果
        suggested_lr = result['suggested_lr'] if isinstance(result, dict) else result
        
        print(f"\n{'='*70}")
        print(f"✅ LR Finder completed!")
        print(f"{'='*70}")
        print(f"   Suggested max_lr: {suggested_lr:.2e}")
        print(f"   Iterations:      {result.get('num_iter', 'N/A') if isinstance(result, dict) else 'N/A'}")
        print(f"   Batch size:      {result.get('batch_size', 'N/A') if isinstance(result, dict) else 'N/A'}")
        print(f"\n   Run with:")
        print(f"   python benchmark_dti.py --device {args.device} --lr {suggested_lr:.2e} ...")
        print(f"{'='*70}\n")
        exit(0)
    
    # 打印关键参数（确认）
    print(f"\n{'='*70}")
    print(f"Key Training Parameters:")
    print(f"{'='*70}")
    print(f"  Use precomputed features:  {args.use_precomputed_features}")
    print(f"{'='*70}")
    
    os.makedirs('checkpoint/DTI/', exist_ok=True)
    os.makedirs('log/DTI/detail', exist_ok=True)
    args.min_epochs = 30
    # 保持命令行参数的值，如果没有设置则使用默认值15
    # args.patience = 15  # 不再硬编码，使用命令行参数
    # args.lr 现在从命令行参数获取，不再硬编码
    np.random.seed(666779)
    
    # 打印关键参数（确认）
    print(f"\n{'='*70}")
    print(f"Key Training Parameters:")
    print(f"{'='*70}")
    print(f"  Patience (early stop):     {args.patience}")
    print(f"  LR reduce rate:            {args.lr_reduce_rate}")
    print(f"  LR reduce patience:        {args.lr_reduce_patience}")
    print(f"  Num workers:               {args.num_workers}")
    print(f"  Batch size:                {args.batch_size}")
    print(f"  Learning rate:             {args.lr}")
    print(f"  Weight decay:              {args.decay}")
    print(f"  Protein extractor:         {args.protein_extractor}")
    if args.protein_extractor in ['esm', 'esm_cnn']:
        print(f"  ESM model:                 {args.esm_model}")
    print(f"  Dataset:                   {args.dataset}")
    print(f"  Split:                     {args.split}")
    print(f"{'='*70}\n")
    
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)  
    args.config = json.dumps(config)
    args.mode = 'cls'
    args.monitor = 'rmse' if args.mode == 'reg' else 'roc'
    
    # 加载已有的结果（如果存在）
    metric_list = []
    test_results = []
    test_results_list = []  # 保持兼容性
    # 在文件名中包含蛋白质提取器类型
    log_file = f'log/DTI/{args.dataset}_{args.split}_{args.protein_extractor}.txt'
    npy_file = f'log/DTI/{args.dataset}_{args.split}_{args.protein_extractor}.npy'
    
    # 从npy文件加载test_results
    if os.path.exists(npy_file):
        test_results = np.load(npy_file, allow_pickle=True).tolist()
    
    # 从log文件加载metric_list
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('=================') and 'run' in line:
                # 下一行应该是 metric 值
                if i + 1 < len(lines):
                    try:
                        metric = float(lines[i + 1].strip())
                        # 确保不是统计行（包含 +/- 的行）
                        if '+/-' not in lines[i + 1]:
                            metric_list.append(metric)
                    except ValueError:
                        pass
            i += 1
        
        print(f"Loaded {len(metric_list)} completed runs from log file")
    
    # 🔧 新增：如果 log 文件中没有记录，但 checkpoint 文件存在，也认为已训练完成
    # 检查是否有 checkpoint 文件存在
    if len(metric_list) < args.runs:
        for run_id in range(1, args.runs + 1):
            # 检查 checkpoint 是否存在
            base_save_path = f'checkpoint/DTI/{args.dataset}_{args.split}_{args.protein_extractor}_lr{args.lr}_bs{args.batch_size}_run{run_id}_global_best_roc.pt'
            if os.path.exists(base_save_path) and (run_id - 1) not in metric_list and run_id not in metric_list:
                # checkpoint 存在但 log 中没有记录，认为该 run 已完成
                if run_id <= len(metric_list):
                    # 确保列表足够长
                    while len(metric_list) < run_id:
                        metric_list.append(None)
                else:
                    metric_list.append(0.0)  # 占位
                print(f"Found checkpoint for run {run_id}, will load and evaluate")
        
        if len(metric_list) > 0:
            print(f"Total completed runs (including checkpoints): {len(metric_list)}")
    
    # 计算实际的run范围
    if args.run_end == -1:
        args.run_end = args.runs
    
    # 自动调整run_start，避免重复运行已完成的run（除非指定了--overwrite）
    completed_runs = len(metric_list)
    if args.run_start <= completed_runs and not args.overwrite:
        print(f"WARNING: run_start ({args.run_start}) <= completed runs ({completed_runs})")
        print(f"Auto-adjusting run_start to {completed_runs + 1}")
        print("Use --overwrite to force re-running completed runs")
        args.run_start = completed_runs + 1
    elif args.run_start <= completed_runs and args.overwrite:
        print(f"WARNING: run_start ({args.run_start}) <= completed runs ({completed_runs})")
        print(f"Overwriting completed runs as requested (--overwrite flag set)")
    
    # 确保run_end不超过args.runs
    if args.run_end > args.runs:
        args.run_end = args.runs
    
    # 确保run_start <= run_end
    if args.run_start > args.run_end:
        # 🔧 对于 test_only 模式，允许这种情况继续执行（用于收集已有结果）
        if getattr(args, 'test_only', False):
            print(f"⚠️  test_only mode: all runs already completed, will collect results from existing checkpoints")
        else:
            print(f"ERROR: run_start ({args.run_start}) > run_end ({args.run_end})")
            print(f"No runs to execute. Exiting...")
            exit(1)
    
    print(f"Starting runs {args.run_start} to {args.run_end} with batch_size={args.batch_size}")
    print(f"Total runs specified: {args.runs}")
    print(f"Loaded {completed_runs} completed runs from existing files")
    print(f"Will execute {args.run_end - args.run_start + 1} new runs")
    
    # 在开始运行前，如果是覆盖模式，清理日志文件
    if args.overwrite and args.run_start == 1:
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"Overwrite mode: removed existing log file: {log_file}")
        if os.path.exists(npy_file):
            os.remove(npy_file)
            print(f"Overwrite mode: removed existing npy file: {npy_file}")
        metric_list = []
        test_results = []
        print("Overwrite mode: cleared existing files")
    
    # 准备要运行的实验列表
    # 2) 优化种子生成：O(n) 而非 O(n²)，续跑/并行跑都一致
    np.random.seed(666779)
    all_seeds = [np.random.randint(0, 100000) for _ in range(args.runs)]
    
    experiments = []
    
    # 🔧 test_only 模式特殊处理：强制遍历所有 run
    if getattr(args, 'test_only', False) and args.run_start > args.run_end:
        print(f"⚠️  test_only mode: will test all {args.runs} runs from checkpoints")
        for i in range(args.runs):
            seed = all_seeds[i]
            experiments.append((i+1, seed))
    else:
        for i in range(args.run_start-1, args.run_end):
            # 检查当前run是否已经完成（除非设置了--overwrite）
            if i < len(metric_list) and not args.overwrite:
                print(f"Skipping Run {i+1}, already completed")
                continue
            # 直接使用预生成的种子
            seed = all_seeds[i]
            experiments.append((i+1, seed))
    
    if args.parallel:
        # 多卡并行运行
        print(f"\n{'='*80}")
        print(f"Running experiments in parallel using multiple GPUs")
        print(f"{'='*80}\n")
        
        # 解析GPU列表
        gpu_ids = [int(gpu) for gpu in args.gpus.split(',')]
        num_gpus = len(gpu_ids)
        print(f"Using GPUs: {gpu_ids}")
        print(f"Total experiments to run: {len(experiments)}")
        
        # 使用Manager来共享返回值
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # 任务队列模式：维护可用GPU池和运行中的进程
        from collections import deque
        import threading
        
        available_gpus = deque(gpu_ids)  # 可用GPU队列
        running_processes = []  # 存储正在运行的进程信息：(process, gpu_id, run_id, seed)
        experiment_queue = deque(experiments)  # 待运行的实验队列
        completed_count = 0
        total_experiments = len(experiments)
        
        print(f"\n{'='*60}")
        print(f"Task Queue Mode: {total_experiments} experiments, {num_gpus} GPUs")
        print(f"{'='*60}\n")
        
        while experiment_queue or running_processes:
            # 1. 检查并清理已完成的进程
            i = 0
            while i < len(running_processes):
                process, gpu_id, run_id, seed = running_processes[i]
                if not process.is_alive():
                    # 进程已完成，回收GPU
                    process.join()
                    available_gpus.append(gpu_id)
                    running_processes.pop(i)
                    completed_count += 1
                    print(f"\n[Progress] Run {run_id} completed | {completed_count}/{total_experiments}")
                else:
                    i += 1
            
            # 2. 分配新任务到可用GPU
            while available_gpus and experiment_queue:
                gpu_id = available_gpus.popleft()
                run_id, seed = experiment_queue.popleft()
                
                print(f"Assigning Run {run_id} to GPU {gpu_id}")
                
                process = mp.Process(
                    target=run_single_experiment,
                    args=(run_id, args, seed, args.batch_size, gpu_id, return_dict)
                )
                running_processes.append((process, gpu_id, run_id, seed))
                process.start()
                
                # 避免进程启动过快
                time.sleep(2)
            
            # 3. 短暂休眠，避免CPU占用过高
            if running_processes:
                time.sleep(1)
        
        print(f"\n{'='*60}")
        print(f"All experiments completed!")
        print(f"{'='*60}\n")
        
        # 收集结果并保存
        for run_id, seed in experiments:
            if run_id in return_dict and return_dict[run_id]['success']:
                result = return_dict[run_id]
                
                # 确保列表足够长
                while len(metric_list) < run_id:
                    metric_list.append(None)
                    test_results.append(None)
                
                # 更新结果 - 保存完整结果字典（包含 full_results 用于结构化导出）
                metric_list[run_id-1] = result['metric']
                test_results[run_id-1] = result  # 保存完整字典，不只是 results
                
                # 过滤掉 None 值来计算统计信息
                valid_metrics = [m for m in metric_list if m is not None]
                metric_np = np.array(valid_metrics)
                
                # 追加模式写入日志
                with open(log_file, 'a') as f:
                    f.write(f"================= {run_id} run {seed} =================\n")
                    f.write("{:.4f}\n".format(result['metric']))
                    # 记录最佳 epoch 和阈值信息（从轻量字段读取）
                    if 'best_epoch_global' in result:
                        f.write(f"Best epoch (global): {result['best_epoch_global']}\n")
                    if 'best_epoch_perprot' in result:
                        f.write(f"Best epoch (per-prot): {result['best_epoch_perprot']}\n")
                    # 1) 从轻量字段读取阈值信息（不再读 full_results）
                    if 'threshold_f1' in result:
                        f.write(f"Threshold F1: {result['threshold_f1']:.4f}\n")
                    if 'threshold_youden' in result:
                        f.write(f"Threshold Youden: {result['threshold_youden']:.4f}\n")
                    if 'valid_positive_rate' in result:
                        f.write(f"Valid positive rate: {result['valid_positive_rate']:.4f}\n")
                    if len(valid_metrics) > 0:
                        f.write("{:.4f} +/- {:.4f}\n".format(metric_np.mean(), metric_np.std()))
                
                # 只保存有效的结果，避免 inhomogeneous shape 错误
                # 从完整字典中提取 results 列表用于保存
                valid_test_results_for_save = []
                for r in test_results:
                    if r is not None:
                        if isinstance(r, dict) and 'results' in r:
                            valid_test_results_for_save.append(r['results'])
                        else:
                            valid_test_results_for_save.append(r)
                if valid_test_results_for_save:
                    np.save(npy_file, np.array(valid_test_results_for_save))
                else:
                    # 如果没有有效结果，不保存
                    print("No valid test results to save")
                
                print(f"\n[Summary] Run {run_id} completed successfully")
                if len(valid_metrics) > 0:
                    print(f"[Summary] Current average {args.monitor}: {metric_np.mean():.4f} +/- {metric_np.std():.4f}")
            else:
                print(f"\n[Summary] Run {run_id} failed, skipping...")
                if run_id in return_dict:
                    print(f"[Summary] Error: {return_dict[run_id].get('error', 'Unknown error')}")
    else:
        # 单卡串行运行
        for run_id, seed in experiments:
            print(f"\n{'='*80}")
            print(f"Starting Run {run_id}/{args.runs} with seed {seed}")
            print(f"{'='*80}\n")
            
            # 使用 Manager 来共享返回值
            manager = mp.Manager()
            return_dict = manager.dict()
            
            # 在子进程中运行
            process = mp.Process(
                target=run_single_experiment,
                args=(run_id, args, seed, args.batch_size, args.device, return_dict)
            )
            process.start()
            process.join(timeout=3600)  # 设置超时时间，避免无限期等待
            
            # 检查进程是否正常结束
            if process.is_alive():
                print(f"Process {process.pid} is still alive after timeout")
                process.terminate()
                process.join()
                return_dict[run_id] = {'success': False, 'error': 'Process timeout'}
            
            # 检查结果
            if run_id in return_dict and return_dict[run_id]['success']:
                result = return_dict[run_id]
                
                # 确保列表足够长
                while len(metric_list) < run_id:
                    metric_list.append(None)
                    test_results.append(None)
                
                # 更新结果 - 保存完整结果字典（包含 full_results 用于结构化导出）
                metric_list[run_id-1] = result['metric']
                test_results[run_id-1] = result  # 保存完整字典，不只是 results
            else:
                print(f"Run {run_id} failed or did not return result")
            
            # 过滤掉 None 值来计算统计信息
            valid_metrics = [m for m in metric_list if m is not None]
            metric_np = np.array(valid_metrics)
            
            # 追加模式写入日志
            with open(log_file, 'a') as f:
                f.write(f"================= {run_id} run {seed} =================\n")
                if run_id in return_dict and return_dict[run_id]['success']:
                    f.write("{:.4f}\n".format(result['metric']))
                    # 记录最佳 epoch 和阈值信息（从轻量字段读取）
                    if 'best_epoch_global' in result:
                        f.write(f"Best epoch (global): {result['best_epoch_global']}\n")
                    if 'best_epoch_perprot' in result:
                        f.write(f"Best epoch (per-prot): {result['best_epoch_perprot']}\n")
                    # 1) 从轻量字段读取阈值信息（不再读 full_results）
                    if 'threshold_f1' in result:
                        f.write(f"Threshold F1: {result['threshold_f1']:.4f}\n")
                    if 'threshold_youden' in result:
                        f.write(f"Threshold Youden: {result['threshold_youden']:.4f}\n")
                    if 'valid_positive_rate' in result:
                        f.write(f"Valid positive rate: {result['valid_positive_rate']:.4f}\n")
                if len(valid_metrics) > 0:
                    f.write("{:.4f} +/- {:.4f}\n".format(metric_np.mean(), metric_np.std()))
            
            # 只保存有效的结果，避免 inhomogeneous shape 错误
            # 从完整字典中提取 results 列表用于保存
            valid_test_results_for_save = []
            for r in test_results:
                if r is not None:
                    if isinstance(r, dict) and 'results' in r:
                        valid_test_results_for_save.append(r['results'])
                    else:
                        valid_test_results_for_save.append(r)
            if valid_test_results_for_save:
                try:
                    np.save(npy_file, np.array(valid_test_results_for_save))
                    print(f"Valid test results saved to: {npy_file}")
                except Exception as e:
                    print(f"Failed to save valid test results: {e}")
            else:
                # 如果没有有效结果，不保存
                print("No valid test results to save")
            
            if run_id in return_dict and return_dict[run_id]['success']:
                print(f"\n[Summary] Run {run_id} completed successfully")
                if len(valid_metrics) > 0:
                    print(f"[Summary] Current average {args.monitor}: {metric_np.mean():.4f} +/- {metric_np.std():.4f}")
            else:
                print(f"\n[Summary] Run {run_id} failed, skipping...")
                if run_id in return_dict:
                    print(f"[Summary] Error: {return_dict[run_id].get('error', 'Unknown error')}")
    
    # 最终统计
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"Final Summary")
    print(f"{'='*80}")
    print(f"{'='*80}\n")

    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Protein Extractor: {args.protein_extractor}")
    if args.protein_extractor in ['esm', 'esm_cnn']:
        print(f"ESM Model: {args.esm_model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print()

    # 过滤 None 值
    valid_metrics = [m for m in metric_list if m is not None]
    valid_test_results_full = [r for r in test_results if r is not None]  # 完整结果字典
    
    # 3) 按 run_id 排序，更利于复查（续跑/并行时顺序可能乱）
    valid_test_results_full = sorted(valid_test_results_full, key=lambda x: x.get('run_id', 10**9))
    
    # 提取旧格式的 results 列表用于兼容
    valid_test_results = []
    for r in valid_test_results_full:
        if isinstance(r, dict) and 'results' in r:
            valid_test_results.append(r['results'])
        else:
            valid_test_results.append(r)

    print(f"Runs requested: {args.runs}")
    print(f"Runs completed: {len(valid_metrics)}")
    print(f"Runs failed: {args.runs - len(valid_metrics)}")
    print()

    if len(valid_metrics) > 0:
        metric_np = np.array(valid_metrics)
        print(f"Performance Metrics:")
        print(f"  {args.monitor.upper()}: {metric_np.mean():.4f} ± {metric_np.std():.4f}")
        print(f"  Min: {metric_np.min():.4f}")
        print(f"  Max: {metric_np.max():.4f}")
        print()
        
        # 格式化并打印评估结果表格
        formatted_table, _ = format_evaluation_results(valid_test_results)
        print(f"Evaluation Results (Mean ± Std)")
        print(f"{'='*70}")
        print(f"Dataset: {args.dataset} | Split: {args.split} | Protein Extractor: {args.protein_extractor} | Runs: {len(valid_metrics)}")
        print(f"{'='*70}")
        print(formatted_table)
        print()
        
        # 显示 EF 结果
        print(f"Enrichment Factor (EF) Results:")
        print(f"{'='*90}")
        print(f"{'Run':<6} {'Seed':<10} {'ROC-EF@0.5%':<15} {'ROC-EF@1%':<15} {'ROC-EF@2%':<15} {'EF-EF@1%':<15}")
        print(f"{'-'*90}")
        
        # 收集所有 EF 结果用于计算均值
        all_ef_results = {
            'roc_ef_0p5pct': [],
            'roc_ef_1pct': [],
            'roc_ef_2pct': [],
            'ef_ef_1pct': [],
            'scafdedup_ef_1pct': []
        }
        
        for r in valid_test_results_full:
            if isinstance(r, dict):
                run_id = r.get('run_id', 'N/A')
                seed = r.get('seed', 'N/A')
                
                # 从 detail.json 读取 EF 结果
                # 使用 glob 模式匹配，因为文件名包含 lr 信息
                detail_pattern = f'log/DTI/detail/{args.dataset}_{args.split}_{args.protein_extractor}/*_run{run_id}_seed{seed}_detail.json'
                detail_files = list(Path('.').glob(detail_pattern))
                
                if detail_files:
                    detail_file = detail_files[0]  # 取第一个匹配的文件
                    try:
                        with open(detail_file, 'r') as f:
                            detail_data = json.load(f)
                        
                        # 提取 EF 结果
                        roc_ef = detail_data.get('global_roc', {}).get('ef', {})
                        ef_ef = detail_data.get('global_ef', {}).get('ef', {})
                        scaf_ef = detail_data.get('scaffold_dedup', {}).get('ef_scaffold_dedup', {})
                        
                        roc_ef_0p5 = roc_ef.get('ef_0p5pct', 0)
                        roc_ef_1 = roc_ef.get('ef_1pct', 0)
                        roc_ef_2 = roc_ef.get('ef_2pct', 0)
                        ef_ef_1 = ef_ef.get('ef_1pct', 0)
                        scaf_ef_1 = scaf_ef.get('ef_1pct_scafdedup', 0)
                        
                        # 收集用于计算均值
                        all_ef_results['roc_ef_0p5pct'].append(roc_ef_0p5)
                        all_ef_results['roc_ef_1pct'].append(roc_ef_1)
                        all_ef_results['roc_ef_2pct'].append(roc_ef_2)
                        all_ef_results['ef_ef_1pct'].append(ef_ef_1)
                        all_ef_results['scafdedup_ef_1pct'].append(scaf_ef_1)
                        
                        print(f"{run_id:<6} {seed:<10} {roc_ef_0p5:<15.4f} {roc_ef_1:<15.4f} {roc_ef_2:<15.4f} {ef_ef_1:<15.4f}")
                    except Exception as e:
                        print(f"{run_id:<6} {seed:<10} {'Error reading EF':<15}")
                else:
                    print(f"{run_id:<6} {seed:<10} {'No detail file':<15}")
        
        # 显示 EF 均值
        if all_ef_results['roc_ef_1pct']:
            print(f"{'-'*90}")
            mean_roc_ef_0p5 = np.mean(all_ef_results['roc_ef_0p5pct'])
            mean_roc_ef_1 = np.mean(all_ef_results['roc_ef_1pct'])
            mean_roc_ef_2 = np.mean(all_ef_results['roc_ef_2pct'])
            mean_ef_ef_1 = np.mean(all_ef_results['ef_ef_1pct'])
            
            std_roc_ef_0p5 = np.std(all_ef_results['roc_ef_0p5pct'])
            std_roc_ef_1 = np.std(all_ef_results['roc_ef_1pct'])
            std_roc_ef_2 = np.std(all_ef_results['roc_ef_2pct'])
            std_ef_ef_1 = np.std(all_ef_results['ef_ef_1pct'])
            
            print(f"{'Mean':<6} {'':<10} {mean_roc_ef_0p5:.4f}±{std_roc_ef_0p5:.4f}   {mean_roc_ef_1:.4f}±{std_roc_ef_1:.4f}   {mean_roc_ef_2:.4f}±{std_roc_ef_2:.4f}   {mean_ef_ef_1:.4f}±{std_ef_ef_1:.4f}")
        
        print(f"{'='*90}\n")
        
        # 显示每个 run 的 best_epoch 信息
        print(f"Best Epochs per Run:")
        print(f"{'='*70}")
        print(f"{'Run':<6} {'Seed':<10} {'Best Epoch (ROC)':<18} {'Best Epoch (EF)':<18} {'Metric':<10}")
        print(f"{'-'*70}")
        for r in valid_test_results_full:
            if isinstance(r, dict):
                run_id = r.get('run_id', 'N/A')
                seed = r.get('seed', 'N/A')
                best_epoch_roc = r.get('best_epoch_global', 'N/A')
                best_epoch_ef = r.get('best_epoch_perprot', 'N/A')
                metric = r.get('metric', 'N/A')
                if metric is not None:
                    metric_str = f"{metric:.4f}"
                else:
                    metric_str = 'N/A'
                print(f"{run_id:<6} {seed:<10} {best_epoch_roc:<18} {best_epoch_ef:<18} {metric_str:<10}")
        print(f"{'='*70}\n")
        
        # 保存评估结果到文件
        txt_file, json_file = save_evaluation_results(args, valid_test_results)
        print(f"Evaluation results saved to:")
        print(f"  - {txt_file} (Text format)")
        print(f"  - {json_file} (JSON format)")
        
        # 保存结构化结果（用于模型对比）
        # 2.1) 不再外部传入 seeds，函数内部直接使用 result['seed']
        structured_json, structured_csv = save_structured_results(args, valid_test_results_full)
        print(f"Structured results saved to:")
        print(f"  - {structured_json} (JSON format)")
        print(f"  - {structured_csv} (CSV format)")
        
        # 2.2) 额外保存 runs.json，包含 run_id/seed/阈值信息（续跑/复查不依赖 log 解析）
        runs_info = []
        for r in valid_test_results_full:
            if isinstance(r, dict):
                runs_info.append({
                    'run_id': r.get('run_id'),
                    'seed': r.get('seed'),
                    'metric': r.get('metric'),
                    'best_epoch_global': r.get('best_epoch_global'),
                    'best_epoch_perprot': r.get('best_epoch_perprot'),
                    'threshold_f1': r.get('threshold_f1'),
                    'threshold_youden': r.get('threshold_youden'),
                    'valid_positive_rate': r.get('valid_positive_rate'),
                    'results': r.get('results'),  # 8值列表
                })
        runs_json_file = f'log/DTI/{args.dataset}_{args.split}_{args.protein_extractor}_runs.json'
        with open(runs_json_file, 'w') as f:
            json.dump(runs_info, f, indent=2)
        print(f"Runs info saved to: {runs_json_file}")
        
        # 保存过滤后的有效结果
        np.save(npy_file, np.array(valid_test_results))
        print(f"Raw results saved to: {npy_file}")
        print(f"Log saved to: {log_file}")
        
        # 直接保存到 log/DTI 目录下
        
        # 生成带有运行次数的文件名
        done_txt_file = f'log/DTI/{args.dataset}_{args.split}_{args.protein_extractor}.txt'
        
        # 收集run详细信息（从日志文件中读取）
        run_details = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('=================') and 'run' in line:
                        # 提取run信息
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                run_id = int(parts[1])
                                seed = int(parts[-2])
                                # 下一行是metric
                                if i + 1 < len(lines):
                                    metric_line = lines[i + 1].strip()
                                    if '+/-' not in metric_line:
                                        try:
                                            metric = float(metric_line)
                                            run_details.append((run_id, seed, metric))
                                        except ValueError:
                                            pass
                            except ValueError:
                                pass
                    i += 1
        
        # 写入结果到 log/DTI 目录（恢复旧格式）
        with open(done_txt_file, 'w') as f:
            # 写入每个run的详细信息
            for run_id, seed, metric in run_details:
                f.write(f"================= {run_id} run {seed} =================\n")
                f.write(f"{metric:.4f}\n")
                # 计算当前的平均值
                current_metrics = [m for r, s, m in run_details if r <= run_id]
                if current_metrics:
                    mean_metric = sum(current_metrics) / len(current_metrics)
                    std_metric = 0.0 if len(current_metrics) == 1 else np.std(current_metrics)
                    f.write(f"{mean_metric:.4f} +/- {std_metric:.4f}\n")
            
            # 写入最终总结
            f.write(f"\n{'='*80}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Final Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Protein Extractor: {args.protein_extractor}\n")
            if args.protein_extractor in ['esm', 'esm_cnn']:
                f.write(f"ESM Model: {args.esm_model}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Patience: {args.patience}\n")
            f.write(f"\n")
            f.write(f"Runs requested: {args.runs}\n")
            f.write(f"Runs completed: {len(valid_metrics)}\n")
            f.write(f"Runs failed: {args.runs - len(valid_metrics)}\n")
            f.write(f"\n")
            if len(valid_metrics) > 0:
                f.write(f"Performance Metrics:\n")
                f.write(f"  {args.monitor.upper()}: {metric_np.mean():.4f} ± {metric_np.std():.4f}\n")
                f.write(f"  Min: {metric_np.min():.4f}\n")
                f.write(f"  Max: {metric_np.max():.4f}\n")
        
        print(f"Compatibility file saved to: {done_txt_file}")
        print(f"Old format (run details) restored!")
    else:
        print("No successful runs to report.")
    
    # 运行时间统计
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTotal execution time: {hours}h {minutes}m {seconds}s")
    print(f"{'='*80}\n")