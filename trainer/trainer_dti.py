# -*- coding: utf-8 -*-
"""
trainer_dti.py
Created on Tue Apr 11 14:18:37 2023

@author: Fanding Xu
"""

import os
import time
import torch
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score, confusion_matrix, roc_curve
from collections import defaultdict
from rdkit import Chem
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, Image

import sys
sys.path.append('../')

from dataset.databuild import from_smiles
from utils import mol_with_atom_index, comps_visualize_multi

class DTITrainer():
    def __init__(self, args, model, device):
        self.args = args
        if args.patience == 0:
            args.epochs = args.min_epochs
        self.model = model.to(device)
        self.device = device
        self.mode = args.mode
        
        # ✅ 梯度裁剪配置
        self.use_grad_clip = True  # 默认启用
        self.max_grad_norm = 1.0
        
        # ✅ 混合精度配置
        self.use_amp = False  # 默认关闭（避免数值不稳定）
        
        # ✅ 保持进度条显示
        self.show_progress = True
        if os.getenv('HIDE_PROGRESS_BAR', '0') == '1':
            self.show_progress = False
        
        if self.mode == 'reg':
            self.val_all = self.__val_reg
            self.loss_function = torch.nn.MSELoss() 
        elif self.mode == 'cls':
            self.val_all = self.__val_cls
            self.loss_function = torch.nn.BCEWithLogitsLoss()
        else: 
            raise ValueError(f"Unknown mode: {self.mode}")
        
        weight_decay = 5e-4 if args.decay == 0 else args.decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-6
        )
        print(f"✅ Using AdamW optimizer (lr={args.lr:.2e}, eps=1e-6)")
        
        # ✅ scheduler 延迟到 __call__ 初始化（需要知道 steps_per_epoch）
        self.scheduler = None
        
        # ✅ 修复残留：current_step 保留，供 debug_mask 使用，支持多次调用 __call__
        self.current_step = 0
        
        print(f"✅ Will use OneCycleLR scheduler (initialized after loader is ready)")
        print(f"   - max_lr: {args.lr:.2e}")
        print(f"   - Initial lr will be: {args.lr / 100:.2e} (max_lr / div_factor)")
        print(f"   - Gradient clipping: {'enabled' if self.use_grad_clip else 'disabled'} (max_norm={self.max_grad_norm})")
        print(f"   - Mixed precision: {'enabled' if self.use_amp else 'disabled'}")
        
        # ✅ 混合精度 scaler
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(enabled=True)
        
        self.best_model_path = None
        self.best_metric = -float('inf')
        self.best_perprot_metric = -float('inf')
        self.best_perprot_epoch = 0
    
    def _unpack_protein(self, t):
        """
        解析 protein 输入，返回 (t, mask, use_precomputed)
        统一处理三种模式：features / input_ids / tokens
        """
        mask = None
        use_precomputed = False
        
        if isinstance(t, dict):
            if 'features' in t:
                use_precomputed = True
                mask = t['attention_mask'].to(self.device)
                t = t['features'].to(self.device)
                # ===== 🔑 CRITICAL: features 维度校验 =====
                if t.dim() != 3:
                    raise ValueError(f"features must be 3D [B, L, D], got {t.shape}")
            elif 'input_ids' in t:
                use_precomputed = False
                mask = t['attention_mask'].to(self.device)
                t = t['input_ids'].to(self.device)
            elif 'tokens' in t:
                use_precomputed = False
                mask = t['attention_mask'].to(self.device)
                t = t['tokens'].to(self.device)
            else:
                raise KeyError(f"Unknown protein dict keys: {list(t.keys())}")
            
            # ===== 🔑 CRITICAL: mask dtype/shape 保险 =====
            if mask is not None:
                mask = (mask > 0).long()
                if mask.dim() != 2:
                    raise ValueError(f"attention_mask must be 2D [B, L], got {mask.shape}")
        else:
            t = t.to(self.device)
        
        return t, mask, use_precomputed
    
    def load_buffer(self, load_path):
        assert os.path.exists(load_path), "load_path does not exist"
        # 兼容两种 checkpoint 格式：
        # 1. 直接保存 state_dict: model.state_dict()
        # 2. 保存为 dict: {'model_state_dict': model.state_dict(), ...}
        ckpt = torch.load(load_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        print(f"********** Model Loaded {load_path} **********")
    
    @staticmethod
    def worker_init_fn(worker_id):
        """确保DataLoader的多进程也能复现"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def _extract_pred(self, model_output):
        """
        统一提取预测值。
        约定：tuple 输出时，预测值始终在 index 0（如 (pred, aux_loss) 或 (pred, attn)）。
        如果模型返回单值，直接返回。
        """
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output
    
    def find_lr(self, loader_tr, start_lr=1e-7, end_lr=1, num_iter=None, 
                save_plot_path=None, smooth_method='savgol', suggest_methods='all',
                num_validation_epochs=3, auto_range=True):
        """
        使用 LR Finder 快速找到合适的学习率范围。
        基于 Smith (2017) 的 Cyclical Learning Rates 方法，增强版。
        
        改进点：
        1. 自动调整迭代次数（基于数据集大小）
        2. 多种平滑方法（EMA + Savitzky-Golay）
        3. 返回多个候选 lr（最大负梯度、最小 loss、发散前点等）
        4. 更鲁棒的梯度计算（滑动窗口平均）
        5. 快速验证建议的 lr

        Args:
            loader_tr:             训练数据加载器
            start_lr:              起始学习率（默认 1e-7）
            end_lr:                结束学习率（默认 1）
            num_iter:              迭代次数（默认 None，自动计算）
            save_plot_path:        保存 lr-loss 曲线图的路径（可选）
            smooth_method:         平滑方法：'ema', 'savgol', 'both'（默认 'savgol'）
            suggest_methods:       建议方法：'steepest', 'minimum', 'diverge', 'all'（默认 'all'）
            num_validation_epochs: 快速验证的 epoch 数（默认 3，设为 0 跳过验证）
            auto_range:            是否自动调整搜索范围（默认 True）

        Returns:
            dict: 包含多个建议 lr 和相关信息
        """
        # ── 自动调整参数 ──────────────────────────────────────────────────────
        dataset_size = len(loader_tr.dataset)
        batch_size = loader_tr.batch_size
        
        # 自动计算迭代次数：至少覆盖 2-3 个 epoch，或最少 200 次
        if num_iter is None:
            steps_per_epoch = len(loader_tr)
            num_iter = max(200, min(500, steps_per_epoch * 3))
        
        # 自动调整搜索范围（基于模型参数量）
        if auto_range:
            num_params = sum(p.numel() for p in self.model.parameters())
            if num_params > 50e6:  # 大模型
                end_lr = min(end_lr, 1e-2)
            elif num_params > 10e6:  # 中等模型
                end_lr = min(end_lr, 1e-1)
        
        print(f"\n{'='*70}")
        print(f"LR Finder: Searching for optimal learning rate (Enhanced)")
        print(f"{'='*70}")
        print(f"  Dataset size:       {dataset_size}")
        print(f"  Batch size:         {batch_size}")
        print(f"  Model parameters:   {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        print(f"  Start LR:           {start_lr:.2e}")
        print(f"  End LR:             {end_lr:.2e}")
        print(f"  Iterations:         {num_iter} (auto: {num_iter is None})")
        print(f"  Smooth method:      {smooth_method}")
        print(f"  Suggest methods:    {suggest_methods}")
        print(f"  Validation epochs:  {num_validation_epochs}")
        print(f"{'='*70}\n")

        # ── 保存原始状态（支持多 param_group） ──────────────────────────────
        original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        original_model_state = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                for k, v in self.model.state_dict().items()}
        original_optim_state = self.optimizer.state_dict()

        # 设置初始 lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr

        # 指数增长因子
        lr_factor = (end_lr / start_lr) ** (1.0 / num_iter)

        losses = []
        lrs = []
        best_loss = float('inf')
        best_lr = start_lr

        self.model.train()
        iterator = iter(loader_tr)

        for iteration in range(num_iter):
            # 循环复用 loader
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader_tr)
                batch = next(iterator)

            g, t, y = batch[:3]
            g = g.to(self.device)
            y = y.to(self.device)
            t, mask, batch_use_precomputed = self._unpack_protein(t)

            self.optimizer.zero_grad(set_to_none=True)
            y = y.view(-1)
            if self.mode == 'cls':
                y = y.float()

            model_output = self.model(g, t, mask=mask, 
                                      use_precomputed_features=batch_use_precomputed)
            pred = self._extract_pred(model_output).view(-1)
            loss = self.loss_function(pred, y)

            if not torch.isfinite(loss):
                print(f"\n⚠️  Loss became NaN/Inf at iteration {iteration}, "
                      f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
                break

            loss.backward()
            self.optimizer.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            losses.append(loss.item())
            lrs.append(current_lr)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_lr = current_lr

            # 更新 lr（指数增长）
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_factor

            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"  Iter {iteration+1:3d}/{num_iter}: "
                      f"lr={current_lr:.2e}, loss={loss.item():.4f}")

        # ── 计算建议 lr（多种方法） ──────────────────────────────────────────────
        suggestions = {
            'steepest': None,
            'minimum': None,
            'diverge': None,
            'geometric_mean': None,
            'conservative': None,
        }
        
        if len(losses) > 20:
            # ── 方法 1: EMA 平滑 ────────────────────────────────────────────────
            smoothed_ema = []
            avg_loss = 0.0
            beta = 0.98  # 更激进的平滑
            for i, l in enumerate(losses):
                avg_loss = beta * avg_loss + (1 - beta) * l
                smoothed_ema.append(avg_loss / (1 - beta ** (i + 1)))
            
            # ── 方法 2: Savitzky-Golay 滤波器（更鲁棒） ─────────────────────────
            smoothed_savgol = None
            if smooth_method in ['savgol', 'both']:
                try:
                    from scipy.signal import savgol_filter
                    # 窗口大小：约 10% 的数据点，必须是奇数
                    window = min(51, max(11, len(losses) // 10))
                    if window % 2 == 0:
                        window += 1
                    smoothed_savgol = savgol_filter(losses, window_length=window, polyorder=3)
                except ImportError:
                    print("⚠️  scipy not available, skipping Savitzky-Golay smoothing")
            
            # 选择使用的平滑曲线
            if smooth_method == 'savgol' and smoothed_savgol is not None:
                smoothed = smoothed_savgol
                smooth_name = 'Savitzky-Golay'
            elif smooth_method == 'both' and smoothed_savgol is not None:
                # 取两种方法的平均
                smoothed = [(a + b) / 2 for a, b in zip(smoothed_ema, smoothed_savgol)]
                smooth_name = 'EMA + Savgol (avg)'
            else:
                smoothed = smoothed_ema
                smooth_name = 'EMA (β=0.98)'
            
            # ── 建议 1: 最大负梯度处（loss 下降最快） ───────────────────────────
            skip = max(1, len(smoothed) // 10)  # 跳过前 10%
            
            # 使用滑动窗口平均梯度（更鲁棒）
            window_size = max(5, len(smoothed) // 20)
            gradients = np.gradient(smoothed[skip:])
            
            # 滑动窗口平均
            if len(gradients) > window_size:
                smoothed_gradients = np.convolve(gradients, 
                                                  np.ones(window_size)/window_size, 
                                                  mode='valid')
                steepest_idx = int(np.argmin(smoothed_gradients)) + skip + window_size // 2
            else:
                steepest_idx = int(np.argmin(gradients)) + skip
            
            steepest_idx = min(steepest_idx, len(lrs) - 1)
            suggestions['steepest'] = lrs[steepest_idx]
            
            # ── 建议 2: 最小 loss 处 ───────────────────────────────────────────
            min_loss_idx = int(np.argmin(smoothed))
            suggestions['minimum'] = lrs[min_loss_idx]
            
            # ── 建议 3: 发散前的点（loss 开始上升） ───────────────────────────────
            # 找到 loss 开始持续上升的点
            diverge_idx = None
            for i in range(min_loss_idx + 1, len(smoothed) - 5):
                # 如果连续 5 个点的 loss 都比前一个点大，认为是发散
                if all(smoothed[i+j] > smoothed[i+j-1] for j in range(1, 6)):
                    diverge_idx = i
                    break
            
            if diverge_idx is not None:
                # 取发散前 10% 的位置
                safe_idx = max(min_loss_idx, diverge_idx - max(5, len(smoothed) // 10))
                suggestions['diverge'] = lrs[safe_idx]
            else:
                suggestions['diverge'] = suggestions['minimum']
            
            # ── 建议 4: 几何平均（steepest 和 minimum 的几何平均） ───────────────
            import math
            suggestions['geometric_mean'] = math.sqrt(suggestions['steepest'] * suggestions['minimum'])
            
            # ── 建议 5: 保守估计（steepest 的 1/2） ───────────────────────────────
            suggestions['conservative'] = suggestions['steepest'] / 2
            
            # ── 智能选择最佳 lr ───────────────────────────────────────────────────
            # 优先级：geometric_mean > conservative > diverge > steepest
            # 理由：geometric_mean 平衡了收敛速度和稳定性
            selected_method = 'geometric_mean'
            suggested_lr = suggestions['geometric_mean']
            
            # ── 打印简要建议 ───────────────────────────────────────────────────────
            print(f"\n{'='*70}")
            print(f"LR Finder Results:")
            print(f"{'='*70}")
            print(f"  ✅ Suggested max_lr: {suggested_lr:.2e}")
            print(f"  📊 Method: {smooth_name}")
            print(f"  🎯 Strategy: Geometric mean of steepest & minimum (balanced)")
            print(f"\n  📍 Reference points:")
            print(f"     - Steepest descent: {suggestions['steepest']:.2e} (faster, may be unstable)")
            print(f"     - Conservative:     {suggestions['conservative']:.2e} (slower, more stable)")
            print(f"{'='*70}")

        # ── 恢复原始状态 ──────────────────────────────────────────────────────
        self.model.load_state_dict(original_model_state)
        self.optimizer.load_state_dict(original_optim_state)
        # ✅ Fix 3：按组恢复 lr，支持多 param_group
        for pg, lr in zip(self.optimizer.param_groups, original_lrs):
            pg['lr'] = lr

        # ── 打印最终使用建议 ──────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"💡 How to use:")
        print(f"{'='*70}")
        print(f"  python benchmark_dti.py --lr {suggested_lr:.2e} ...")
        print(f"\n  OneCycleLR will automatically:")
        print(f"    - Start from:  {suggested_lr / 100:.2e} (warmup phase)")
        print(f"    - Peak at:     {suggested_lr:.2e} (max_lr)")
        print(f"    - End at:      {suggested_lr / 10000:.2e} (final_lr)")
        print(f"{'='*70}\n")

        # ── 绘制 lr-loss 曲线（增强版） ──────────────────────────────────────
        if save_plot_path and len(lrs) > 0:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                
                # 上图：loss 曲线
                ax1 = axes[0]
                ax1.plot(lrs, smoothed, label=f'Smoothed ({smooth_name})', 
                        color='steelblue', linewidth=2)
                ax1.plot(lrs, losses, alpha=0.3, color='gray', 
                        linewidth=1, label='Raw Loss')
                ax1.set_xscale('log')
                ax1.set_xlabel('Learning Rate')
                ax1.set_ylabel('Loss')
                ax1.set_title(f'LR Finder: Loss vs Learning Rate (iterations={num_iter})')
                
                # 标记所有建议点
                colors = ['red', 'orange', 'green', 'purple', 'brown']
                labels = ['Steepest', 'Minimum', 'Diverge', 'Geometric', 'Conservative']
                for i, (key, label, color) in enumerate(zip(
                    ['steepest', 'minimum', 'diverge', 'geometric_mean', 'conservative'],
                    labels, colors)):
                    if suggestions[key] is not None:
                        ax1.axvline(x=suggestions[key], color=color, linestyle='--', alpha=0.7,
                                   label=f'{label}: {suggestions[key]:.2e}')
                
                ax1.legend(fontsize=8, loc='upper left')
                ax1.grid(True, alpha=0.3)
                
                # 下图：梯度曲线
                ax2 = axes[1]
                gradients_full = np.gradient(smoothed)
                ax2.plot(lrs, gradients_full, label='Gradient', color='darkgreen', linewidth=2)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_xscale('log')
                ax2.set_xlabel('Learning Rate')
                ax2.set_ylabel('Gradient (dLoss/dLR)')
                ax2.set_title('Gradient of Loss (negative = loss decreasing)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(save_plot_path, dpi=150)
                plt.close(fig)
                print(f"📊 LR-loss curve saved to: {save_plot_path}")
            except ImportError:
                print("⚠️  matplotlib not available, skipping plot")
        
        # ── 快速验证（可选）──────────────────────────────────────────────────────
        if num_validation_epochs > 0:
            print(f"\n{'='*70}")
            print(f"🔍 Quick Validation ({num_validation_epochs} epochs)...")
            print(f"{'='*70}")
            
            validation_lr = suggested_lr
            print(f"Testing lr={validation_lr:.2e}...")
            
            # 临时修改 lr
            original_lrs_temp = [pg['lr'] for pg in self.optimizer.param_groups]
            for pg in self.optimizer.param_groups:
                pg['lr'] = validation_lr
            
            # 快速训练几个 epoch
            val_losses = []
            for epoch in range(num_validation_epochs):
                epoch_loss = 0
                num_batches = 0
                for batch in loader_tr:
                    g, t, y = batch[:3]
                    g = g.to(self.device)
                    y = y.to(self.device)
                    t, mask, batch_use_precomputed = self._unpack_protein(t)
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    y = y.view(-1)
                    if self.mode == 'cls':
                        y = y.float()
                    
                    model_output = self.model(g, t, mask=mask, 
                                             use_precomputed_features=batch_use_precomputed)
                    pred = self._extract_pred(model_output).view(-1)
                    loss = self.loss_function(pred, y)
                    
                    if torch.isfinite(loss):
                        loss.backward()
                        self.optimizer.step()
                        epoch_loss += loss.item()
                        num_batches += 1
                
                avg_loss = epoch_loss / max(num_batches, 1)
                val_losses.append(avg_loss)
                print(f"  Epoch {epoch+1}/{num_validation_epochs}: loss={avg_loss:.4f}")
            
            # 恢复 lr
            for pg, lr in zip(self.optimizer.param_groups, original_lrs_temp):
                pg['lr'] = lr
            
            # ── 智能调整建议 ───────────────────────────────────────────────────────
            if len(val_losses) >= 2:
                loss_change = val_losses[-1] - val_losses[0]
                if val_losses[-1] < val_losses[0]:
                    print(f"\n✅ Validation PASSED: loss decreased by {-loss_change:.4f}")
                    print(f"   Recommended lr: {validation_lr:.2e} (use this)")
                else:
                    # 自动降级到 conservative
                    suggested_lr = suggestions.get('conservative', validation_lr / 2)
                    print(f"\n⚠️  Validation WARNING: loss did not decrease")
                    print(f"   Auto-adjusted to: {suggested_lr:.2e} (more conservative)")
                    print(f"   If still unstable, try: {suggestions.get('conservative', validation_lr / 3):.2e}")
            print(f"{'='*70}\n")
        
        # ── 返回完整结果 ──────────────────────────────────────────────────────
        return {
            'suggested_lr': suggested_lr,
            'suggestions': suggestions,
            'best_loss': best_loss,
            'best_lr': best_lr,
            'lrs': lrs,
            'losses': losses,
            'smoothed': smoothed if len(losses) > 20 else losses,
            'num_iter': num_iter,
            'batch_size': batch_size,
        }
    
    def train_step(self, g, t, y, mask=None, use_precomputed=False):
        """
        单步训练，集成梯度裁剪和混合精度
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        y = y.view(-1)
        if self.mode == 'cls':
            y = y.float()
        
        # ✅ debug_mask：current_step 现在始终存在，不会报错
        if mask is not None and getattr(self.args, 'debug_mask', False) and self.current_step < 3:
            valid_lengths = mask.sum(dim=1)
            print(f"[DEBUG] Step {self.current_step}: mask shape={mask.shape}, "
                  f"valid_len min={valid_lengths.min()}, max={valid_lengths.max()}, "
                  f"mean={valid_lengths.float().mean():.1f}")
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                model_output = self.model(g, t, mask=mask, use_precomputed_features=use_precomputed)
                pred = self._extract_pred(model_output).view(-1)
                loss = self.loss_function(pred, y)
            self.scaler.scale(loss).backward()
            if self.use_grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            model_output = self.model(g, t, mask=mask, use_precomputed_features=use_precomputed)
            pred = self._extract_pred(model_output).view(-1)
            loss = self.loss_function(pred, y)
            loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # ✅ OneCycleLR: 每 step 更新一次
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.current_step += 1
        return loss.item()
    
    def __call__(self, loader_tr, loader_va, loader_te=None,
                save_path='checkpoint/ec50.pt', load_path=None, tensorboard=False):
        """训练模型"""
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"✅ Checkpoint directory created/verified: {save_dir}")
        
        # 确保日志目录存在
        log_dir = "log/DTI/detail"
        os.makedirs(log_dir, exist_ok=True)
        print(f"✅ Log directory created/verified: {log_dir}")
        
        # ✅ 固定所有随机源
        seed = getattr(self.args, 'seed', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # ✅ 初始化 OneCycleLR
        # 用 total_steps 而非 steps_per_epoch + epochs，early stop 时不会崩溃
        steps_per_epoch = len(loader_tr)
        total_steps = steps_per_epoch * self.args.epochs
        
        # ✅ 重置 current_step，支持多次调用 __call__
        self.current_step = 0
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=100.0,
            final_div_factor=1e4,
            three_phase=False,
        )
        
        print(f"\n{'='*70}")
        print(f"Training Configuration:")
        print(f"{'='*70}")
        print(f"  Seed:                      {seed}")
        print(f"  Patience (early stop):     {self.args.patience}")
        print(f"  Monitor metric:            {self.args.monitor}")
        print(f"  Gradient clipping:         {self.use_grad_clip} (max_norm={self.max_grad_norm})")
        print(f"  Mixed precision:           {self.use_amp}")
        print(f"  Save path:                 {save_path}")
        print(f"  Protein extractor:         {getattr(self.args, 'protein_extractor', 'unknown')}")
        print(f"  Dataset:                   {getattr(self.args, 'dataset', 'unknown')}")
        print(f"  Split:                     {getattr(self.args, 'split', 'unknown')}")
        print(f"  --- OneCycleLR ---")
        print(f"  Steps per epoch:           {steps_per_epoch}")
        print(f"  Total steps:               {total_steps}")
        print(f"  Max LR:                    {self.args.lr:.2e}")
        print(f"  Initial LR:                {self.args.lr / 100:.2e}")
        print(f"  Warmup steps:              {int(total_steps * 0.1)} (10%)")
        print(f"{'='*70}\n")
        
        if tensorboard:
            tb = SummaryWriter(log_dir='log/tensorboard')
            note = defaultdict(list)
        
        run_id = getattr(self.args, 'run_id', None)
        device_id = getattr(self.args, 'device_id', None)
        if run_id is not None and device_id is not None:
            self.run_time = f"{self.args.dataset}_{self.args.split}_{self.args.protein_extractor}_run{run_id}_gpu{device_id}"
        else:
            self.run_time = time.strftime("RUN-%Y%m%d-%H%M%S", time.localtime())
        
        with open(f"{log_dir}/{self.run_time}.txt","w") as f:
            f.write(self.args.config+'\n')
        
        if load_path is not None:
            self.load_buffer(load_path)
        
        # ✅ 每次调用 __call__ 都重置，避免多次实验状态污染
        self.best_roc_metric = 0.0
        self.best_roc_epoch = 0
        self.best_ef_metric = -np.inf
        self.best_ef_epoch = 0
        self.best_metric = -float('inf')
        
        # 初始化 best（显式区分 min/max 指标）
        min_metrics = {'loss', 'rmse', 'mse', 'mae'}
        
        if self.args.monitor in min_metrics:
            best = float('inf')
        else:
            best = -float('inf')
        best_epoch = 0
        times = 0
        
        for epoch in range(1, self.args.epochs + 1):
            tic = time.time()
            print("Epoch: {:d}/{:d}".format(epoch, self.args.epochs))
            self.model.train()
            epoch_loss = 0
            
            # 条件显示进度条
            if self.show_progress:
                pbar = tqdm(loader_tr, 
                           desc=f"Training Epoch {epoch}/{self.args.epochs}",
                           miniters=50,
                           mininterval=1.0,
                           ncols=100,
                           file=sys.stdout)
            else:
                pbar = loader_tr
            
            for batch_idx, batch in enumerate(pbar):
                # 支持两种格式：
                # 1. (g, t, y) - 传统格式
                # 2. (g, {'input_ids': t, 'attention_mask': mask}, y) - 新格式
                g, t, y = batch[:3]
                
                g = g.to(self.device)
                y = y.to(self.device)
                
                # 使用统一的 unpack 函数
                t, mask, batch_use_precomputed = self._unpack_protein(t)
                
                # 🔧 使用新的 train_step 方法
                loss = self.train_step(g, t, y, mask=mask, use_precomputed=batch_use_precomputed)
                epoch_loss += loss
                
                # 更新进度条
                if self.show_progress and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
                # scheduler.step() 已在 train_step 内部调用，此处无需重复
            
            loss_tr = epoch_loss / len(loader_tr)
            plr = self.optimizer.param_groups[0]['lr']
            print(f'lr：{round(plr, 7)}')
            print("Training loss = {:.4f}".format(loss_tr))
            
            # 验证
            info_dict = self.val_all(loader_va)
            
            # Tensorboard 记录
            if tensorboard:
                for key, value in info_dict.items():
                    # ✅ 只写入有限的标量值，跳过 nan、int k_ 统计量、非数值
                    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                        tb.add_scalars(key, {'val': float(value)}, epoch)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        note[name+'_param'].append(param.mean())
                        note[name+'_grad'].append(param.grad.mean())
                        tb.add_histogram(name + '_param', param, epoch)
                        tb.add_histogram(name + '_grad', param.grad, epoch)
            
            self.log_info(info_dict)
            state = info_dict[self.args.monitor]
            
            # Note: 单靶点任务跳过 per-protein 指标计算
            
            # ===== 保存逻辑：两套独立的 best（完全解耦） =====
            # save_path 是基础路径（如：.../human_cold_protein_cnn_lr2e-05_bs64_run1）
            
            # 1. 全局最佳 ROC-based（始终基于 ROC，与 args.monitor 无关）
            roc_state = info_dict.get('roc', 0.0)
            
            improved_roc = (roc_state - self.best_roc_metric) > 1e-4
            
            if improved_roc:
                self.best_roc_metric = roc_state
                self.best_roc_epoch = epoch
                
                save_path_global_roc = f"{save_path}_global_best_roc.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_roc': roc_state,
                    'valid_prc': info_dict.get('prc', None),
                    'valid_loss': info_dict.get('loss', None),
                    'best_valid_roc': self.best_roc_metric,
                    'best_valid_metric': 'roc',
                }, save_path_global_roc)
                
                print(f"✅ New best GLOBAL ROC model saved, epoch {self.best_roc_epoch}, ROC-AUC: {self.best_roc_metric:.4f}")
            else:
                if self.best_roc_epoch > 0:
                    print(f"   ROC: {roc_state:.4f} (best: {self.best_roc_metric:.4f} @ epoch {self.best_roc_epoch})")
                else:
                    print(f"   ROC: {roc_state:.4f} (no best yet)")
            
            # 更新基于 args.monitor 的 best（用于 early stopping）
            improved = (best - state) > 1e-4 if self.args.monitor in min_metrics else (state - best) > 1e-4
            if improved:
                best = state
                best_epoch = epoch
                self.best_metric = state
                times = 0
                print(f"✅ New best {self.args.monitor}: {best:.4f} (epoch {best_epoch}, early stopping counter reset)")
            else:
                times += 1
                if best_epoch > 0:
                    print(f"⚠️  {self.args.monitor} not improved for {times} times, current best: {best:.4f} @ epoch {best_epoch}")
                else:
                    print(f"⚠️  {self.args.monitor} not improved for {times} times (no best yet)")
            
            # 2. 全局最佳 EF（用于虚拟筛选）
            # 使用加权组合：0.5% * 0.25 + 1% * 0.5 + 2% * 0.25
            # 这样更稳健，不完全依赖单一比例
            ef_0p5 = info_dict.get('ef_0p5pct', 0.0)
            ef_1 = info_dict.get('ef_1pct', 0.0)
            ef_2 = info_dict.get('ef_2pct', 0.0)
            
            # ✅ nan 保护：将 nan 替换为 0，避免加权组合变成 nan
            ef_0p5 = float(ef_0p5) if np.isfinite(ef_0p5) else 0.0
            ef_1 = float(ef_1) if np.isfinite(ef_1) else 0.0
            ef_2 = float(ef_2) if np.isfinite(ef_2) else 0.0
            
            ef_combined = ef_0p5 * 0.25 + ef_1 * 0.5 + ef_2 * 0.25
            
            ef_metric_name = 'ef_combined'  # 组合 EF
            ef_state = ef_combined
            
            improved_ef = (ef_state - self.best_ef_metric) > 1e-4
            
            if improved_ef:
                self.best_ef_metric = ef_state
                self.best_ef_epoch = epoch
                
                save_path_global_ef = f"{save_path}_global_best_ef_combined.pt"
                # 收集所有 EF 指标
                ef_metrics = {k: v for k, v in info_dict.items() if k.startswith('ef_')}
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_roc': info_dict.get('roc', None),
                    'valid_prc': info_dict.get('prc', None),
                    'valid_ef_metrics': ef_metrics,  # 所有 EF 指标
                    'best_ef_metric': ef_state,
                    'best_ef_metric_name': ef_metric_name,
                }, save_path_global_ef)
                
                print(f"✅ New best GLOBAL EF model saved, epoch {epoch}, {ef_metric_name}: {ef_state:.4f}")
                print(f"   EF breakdown: 0.5%={ef_0p5:.2f}, 1%={ef_1:.2f}, 2%={ef_2:.2f}")
            
            # Note: 单靶点任务不需要 per-protein checkpoint，已移除
            
            # Early stopping
            if epoch > self.args.min_epochs and times >= self.args.patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
            
            toc = time.time()
            print("Time costs: {:.3f}\n".format(toc-tic))
        
        # 训练结束后，加载并测试两套最佳模型
        if loader_te is not None:
            print(f"\n{'='*70}")
            print(f"Testing with BEST GLOBAL model...")
            print(f"{'='*70}\n")
            
            # 2.3 初始化返回结果（单靶点简化版）
            test_results_global = None
            test_results_ef = None
            threshold_info = None
            threshold_info_ef = None
            scaffold_dedup_results = None  # scaffold-dedup 结果
            
            # 1. 测试 Global ROC Best
            save_path_global_roc = f"{save_path}_global_best_roc.pt"
            if os.path.exists(save_path_global_roc):
                checkpoint = torch.load(save_path_global_roc, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    metric = checkpoint.get('best_valid_metric', checkpoint.get('monitor', 'unknown'))
                    bestv = checkpoint.get('best_valid', None)
                    if bestv is not None:
                        print(f"✅ Best GLOBAL model loaded (epoch {checkpoint['epoch']}, best {metric}: {bestv:.4f})")
                    else:
                        print(f"✅ Best GLOBAL model loaded (epoch {checkpoint.get('epoch', '?')})")
                else:
                    self.model.load_state_dict(checkpoint)
                    print(f"✅ Best GLOBAL model loaded")
                
                # 在 valid 集上选择最优阈值（同时计算 F1 和 Youden J 两种阈值）
                print(f"\n🔍 Selecting thresholds on validation set (F1 & Youden J)...")
                threshold_info = self.select_threshold_on_valid(loader_va, criterion='f1', verbose=True)
                
                # 使用 F1 阈值在 test 集上评估
                print(f"\n📊 Evaluating with F1-based threshold (t*={threshold_info['threshold_f1']:.4f})...")
                test_results_f1 = self.test(loader_te, threshold=threshold_info['threshold_f1'])
                
                # 使用 Youden J 阈值在 test 集上评估
                print(f"\n📊 Evaluating with Youden J-based threshold (t*={threshold_info['threshold_youden']:.4f})...")
                test_results_youden = self.test(loader_te, threshold=threshold_info['threshold_youden'])
                
                # 保存两种结果
                test_results_global = {
                    'f1': test_results_f1,
                    'youden': test_results_youden,
                }
                
                if test_results_f1 is not None:
                    # 处理可能包含 EF 字典的返回值（9个元素）
                    if len(test_results_f1) == 9:
                        auroc, auprc, test_positive_rate, acc_f1, sens_f1, spec_f1, prec_f1, _, ef_dict_f1 = test_results_f1
                    else:
                        auroc, auprc, test_positive_rate, acc_f1, sens_f1, spec_f1, prec_f1, _ = test_results_f1
                        ef_dict_f1 = {}
                    print(f"\n📊 GLOBAL Best Test Results:")
                    print(f"\n  Primary Metrics (Threshold-independent):")
                    print(f"    AUROC:          {auroc:.4f}")
                    print(f"    AUPRC:          {auprc:.4f}")
                    print(f"    AUPRC Baseline: {test_positive_rate:.4f} (test positive rate)")
                    
                    # 打印多比例 EF
                    if ef_dict_f1:
                        print(f"\n  Enrichment Factors:")
                        for ef_name, ef_val in ef_dict_f1.items():
                            val_str = f"{ef_val:.2f}" if isinstance(ef_val, (int, float)) and np.isfinite(ef_val) else "nan"
                            print(f"    {ef_name}: {val_str}")
                    
                    print(f"\n  Dataset Shift Check:")
                    print(f"    Valid positive rate: {threshold_info['positive_rate']:.4f}")
                    print(f"    Test positive rate:  {test_positive_rate:.4f}")
                    if abs(threshold_info['positive_rate'] - test_positive_rate) > 0.05:
                        print(f"    ⚠️  Warning: >5% prior shift detected between valid and test")
                    print(f"\n  Secondary Metrics (Threshold-dependent):")
                    print(f"    ┌─ F1-based (t*={threshold_info['threshold_f1']:.4f}):")
                    print(f"    │  Accuracy:    {acc_f1:.4f}")
                    print(f"    │  Sensitivity: {sens_f1:.4f}")
                    print(f"    │  Specificity: {spec_f1:.4f}")
                    print(f"    │  Precision:   {prec_f1:.4f}")
                
                if test_results_youden is not None:
                    # 处理可能包含 EF 字典的返回值（9个元素）
                    if len(test_results_youden) == 9:
                        _, _, _, acc_y, sens_y, spec_y, prec_y, _, _ = test_results_youden
                    else:
                        _, _, _, acc_y, sens_y, spec_y, prec_y, _ = test_results_youden
                    print(f"    └─ Youden J-based (t*={threshold_info['threshold_youden']:.4f}):")
                    print(f"       Accuracy:    {acc_y:.4f}")
                    print(f"       Sensitivity: {sens_y:.4f}")
                    print(f"       Specificity: {spec_y:.4f}")
                    print(f"       Precision:   {prec_y:.4f}")
            
            # Note: 单靶点任务跳过 per-protein 测试
            
            # 2. 测试 Global EF@1% Best（用于虚拟筛选）
            save_path_ef = f"{save_path}_global_best_ef_combined.pt"
            if os.path.exists(save_path_ef):
                print(f"\n{'='*70}")
                print(f"Testing with BEST EF(combined) model (for virtual screening)...")
                print(f"{'='*70}\n")
                
                checkpoint_ef = torch.load(save_path_ef, map_location=self.device)
                if isinstance(checkpoint_ef, dict) and 'model_state_dict' in checkpoint_ef:
                    self.model.load_state_dict(checkpoint_ef['model_state_dict'])
                    ef_metric_name = checkpoint_ef.get('best_ef_metric_name', 'unknown')
                    ef_bestv = checkpoint_ef.get('best_ef_metric', None)
                    if ef_bestv is not None:
                        print(f"✅ Best EF-combined model loaded (epoch {checkpoint_ef['epoch']}, best {ef_metric_name}: {ef_bestv:.4f})")
                    else:
                        print(f"✅ Best EF-combined model loaded (epoch {checkpoint_ef.get('epoch', '?')})")
                else:
                    self.model.load_state_dict(checkpoint_ef)
                    print(f"✅ Best EF-combined model loaded")
                
                # 在 valid 集上选择最优阈值
                print(f"\n🔍 Selecting thresholds on validation set (F1 & Youden J)...")
                threshold_info_ef = self.select_threshold_on_valid(loader_va, criterion='f1', verbose=True)
                
                # 使用两种阈值分别评估
                print(f"\n📊 Evaluating with F1-based threshold (t*={threshold_info_ef['threshold_f1']:.4f})...")
                test_results_ef_f1 = self.test(loader_te, threshold=threshold_info_ef['threshold_f1'])
                
                print(f"\n📊 Evaluating with Youden J-based threshold (t*={threshold_info_ef['threshold_youden']:.4f})...")
                test_results_ef_youden = self.test(loader_te, threshold=threshold_info_ef['threshold_youden'])
                
                test_results_ef = {
                    'f1': test_results_ef_f1,
                    'youden': test_results_ef_youden,
                }
                
                if test_results_ef_f1 is not None:
                    # 处理可能包含 EF 字典的返回值（9个元素）
                    if len(test_results_ef_f1) == 9:
                        auroc, auprc, test_positive_rate, acc_f1, sens_f1, spec_f1, prec_f1, _, ef_dict_ef_print = test_results_ef_f1
                    else:
                        auroc, auprc, test_positive_rate, acc_f1, sens_f1, spec_f1, prec_f1, _ = test_results_ef_f1
                        ef_dict_ef_print = {}
                    
                    print(f"\n📊 EF(combined) Best Test Results:")
                    print(f"\n  Primary Metrics (Threshold-independent):")
                    print(f"    AUROC:          {auroc:.4f}")
                    print(f"    AUPRC:          {auprc:.4f}")
                    print(f"    AUPRC Baseline: {test_positive_rate:.4f} (test positive rate)")
                    
                    # 打印多比例 EF
                    if ef_dict_ef_print:
                        print(f"\n  Enrichment Factors:")
                        for ef_name, ef_val in ef_dict_ef_print.items():
                            val_str = f"{ef_val:.2f}" if isinstance(ef_val, (int, float)) and np.isfinite(ef_val) else "nan"
                            print(f"    {ef_name}: {val_str}")
                    
                    print(f"\n  Dataset Shift Check:")
                    print(f"    Valid positive rate: {threshold_info_ef['positive_rate']:.4f}")
                    print(f"    Test positive rate:  {test_positive_rate:.4f}")
                    if abs(threshold_info_ef['positive_rate'] - test_positive_rate) > 0.05:
                        print(f"    ⚠️  Warning: >5% prior shift detected between valid and test")
                    print(f"\n  Secondary Metrics (Threshold-dependent):")
                    print(f"    ┌─ F1-based (t*={threshold_info_ef['threshold_f1']:.4f}):")
                    print(f"    │  Accuracy:    {acc_f1:.4f}")
                    print(f"    │  Sensitivity: {sens_f1:.4f}")
                    print(f"    │  Specificity: {spec_f1:.4f}")
                    print(f"    │  Precision:   {prec_f1:.4f}")
                
                if test_results_ef_youden is not None:
                    # 处理可能包含 EF 字典的返回值（9个元素）
                    if len(test_results_ef_youden) == 9:
                        _, _, _, acc_y, sens_y, spec_y, prec_y, _, _ = test_results_ef_youden
                    else:
                        _, _, _, acc_y, sens_y, spec_y, prec_y, _ = test_results_ef_youden
                    print(f"    └─ Youden J-based (t*={threshold_info_ef['threshold_youden']:.4f}):")
                    print(f"       Accuracy:    {acc_y:.4f}")
                    print(f"       Sensitivity: {sens_y:.4f}")
                    print(f"       Specificity: {spec_y:.4f}")
                    print(f"       Precision:   {prec_y:.4f}")
                
                # 3. Scaffold-dedup EF 评估（虚拟筛选专用）
                print(f"\n{'='*70}")
                print(f"Scaffold-dedup Evaluation (Virtual Screening)...")
                print(f"{'='*70}")
                vs_results = self.test_with_scaffold_dedup(
                    loader_te, 
                    threshold=threshold_info_ef['threshold_f1'],
                    max_per_scaffold=1
                )
                if vs_results:
                    # 保存 scaffold-dedup 结果供返回
                    scaffold_dedup_results = {
                        'ef_standard': vs_results['ef'],
                        'ef_scaffold_dedup': vs_results['ef_scaffold_dedup'],
                        'scaffold_dedup_config': vs_results['scaffold_dedup_config'],
                        'n_molecules': vs_results['n_molecules'],
                        'n_scaffolds': vs_results['n_scaffolds'],
                    }
            
            if tensorboard:
                tb.close()
            
            # 3) 返回结构化字典，包含两种模型的结果（单靶点简化版）
            # 提取 EF 字典（如果存在）
            ef_dict_global = {}
            ef_dict_ef_final = {}
            if test_results_global and test_results_global.get('f1') and len(test_results_global['f1']) == 9:
                ef_dict_global = test_results_global['f1'][8]  # 第9个元素是 EF 字典
            if test_results_ef and test_results_ef.get('f1') and len(test_results_ef['f1']) == 9:
                ef_dict_ef_final = test_results_ef['f1'][8]
            
            return {
                'global_roc': {
                    'f1': test_results_global['f1'] if test_results_global else None,
                    'youden': test_results_global['youden'] if test_results_global else None,
                    'threshold_f1': threshold_info['threshold_f1'] if threshold_info else 0.5,
                    'threshold_youden': threshold_info['threshold_youden'] if threshold_info else 0.5,
                    'valid_positive_rate': threshold_info['positive_rate'] if threshold_info else 0.0,
                    'ef': ef_dict_global,  # 多比例 EF
                },
                'global_ef': {
                    'f1': test_results_ef['f1'] if test_results_ef else None,
                    'youden': test_results_ef['youden'] if test_results_ef else None,
                    'threshold_f1': threshold_info_ef['threshold_f1'] if threshold_info_ef else 0.5,
                    'threshold_youden': threshold_info_ef['threshold_youden'] if threshold_info_ef else 0.5,
                    'valid_positive_rate': threshold_info_ef['positive_rate'] if threshold_info_ef else 0.0,
                    'ef': ef_dict_ef_final,  # 多比例 EF
                },
                'scaffold_dedup': scaffold_dedup_results,
                'best_epoch_global_roc': getattr(self, 'best_roc_epoch', 0),
                'best_epoch_global_ef': getattr(self, 'best_ef_epoch', 0),
                'best_epoch_monitor': best_epoch,  # args.monitor 的 best epoch
            }
        
        if tensorboard:
            tb.close()
        
        return None
    
    def __val_reg(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        
        if self.show_progress:
            loader_iter = tqdm(loader, desc="validating...", file=sys.stdout)
        else:
            loader_iter = loader
        
        # ✅ Fix: 与 __val_cls 保持一致，兼容任意长度 batch
        for batch in loader_iter:
            g, t, y = batch[:3]
            
            g = g.to(self.device)
            y = y.to(self.device)
            
            t, mask, batch_use_precomputed = self._unpack_protein(t)
            
            with torch.no_grad():
                model_output = self.model(g, t, mask=mask, use_precomputed_features=batch_use_precomputed)
                # ✅ 用 _extract_pred 消除重复的 if/elif 块
                pred = self._extract_pred(model_output).view(-1)
                y = y.view(-1).float()
                loss = self.loss_function(pred, y)
                loss_val += loss.item()
                y_true.append(y)
                y_scores.append(pred)
        
        loss_val /= len(loader)
        
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
        
        y_true = y_true.reshape(-1)
        y_scores = y_scores.reshape(-1)
        
        # ✅ finite 检查：回归任务同样可能出现 nan 预测值
        if not np.isfinite(y_scores).all():
            nan_count = np.sum(~np.isfinite(y_scores))
            print(f"⚠️  检测到 {nan_count} 个非有限值在 reg y_scores 中！")
            raise ValueError("reg y_scores 包含非有限值！")
        
        cur_rmse = root_mean_squared_error(y_true, y_scores)
        
        info_dict = {'loss': loss_val}
        info_dict['rmse'] = float(cur_rmse)
        info_dict['mse'] = float(mean_squared_error(y_true, y_scores))
        info_dict['r2'] = float(r2_score(y_true, y_scores))
        info_dict['mae'] = float(mean_absolute_error(y_true, y_scores))
        return info_dict

    def __val_cls(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        
        if self.show_progress:
            loader_iter = tqdm(loader, desc="validating...", file=sys.stdout)
        else:
            loader_iter = loader
            
        for batch_idx, batch in enumerate(loader_iter):
            g, t, y = batch[:3]
            
            # DEBUG: 在解包后打印（仅 debug_mask 模式）
            if batch_idx == 0 and getattr(self.args, 'debug_mask', False):
                if isinstance(t, dict):
                    print(f"[DEBUG] First batch keys: {list(t.keys())}")
                else:
                    print(f"[DEBUG] First batch: t={t.shape}/{t.dtype}")
            
            g = g.to(self.device)
            y = y.to(self.device)
            
            # 先判断 t 是否是 dict，再搬运到 device
            # 使用统一的 unpack 函数
            t, mask, batch_use_precomputed = self._unpack_protein(t)
            
            with torch.no_grad():
                model_output = self.model(g, t, mask=mask, use_precomputed_features=batch_use_precomputed)
                pred = self._extract_pred(model_output).view(-1)
                y = y.view(-1).float()
                loss = self.loss_function(pred, y)
                loss_val += loss.item()
                y_true.append(y)
                y_scores.append(pred.sigmoid().view(-1))
        
        loss_val /= len(loader)
        
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
        
        # ===== 🔴 额外保险：显式 reshape 成 1D =====
        y_true = y_true.reshape(-1)
        y_scores = y_scores.reshape(-1)
        
        # ===== 🔴 额外保险：提前检查 finite =====
        if not np.isfinite(y_scores).all():
            nan_count = np.sum(~np.isfinite(y_scores))
            print(f"⚠️  检测到 {nan_count} 个非有限值在 y_scores 中！")
            # 打印一些调试信息
            print(f"   - y_scores min: {np.nanmin(y_scores)}, max: {np.nanmax(y_scores)}")
            print(f"   - y_scores mean: {np.nanmean(y_scores)}, std: {np.nanstd(y_scores)}")
            raise ValueError("y_scores 包含非有限值！")
        
        cur_roc = roc_auc_score(y_true, y_scores)
        # 使用 average_precision_score 与 test() 保持一致
        cur_prc = average_precision_score(y_true, y_scores)
        
        # ===== 🔑 计算多比例 EF（用于虚拟筛选评估） =====
        # EF@0.5%, EF@1%, EF@2%
        ef_dict = self._compute_enrichment_factors(y_true, y_scores, [0.005, 0.01, 0.02])
        
        info_dict = {'loss': loss_val}
        info_dict['roc'] = float(cur_roc)
        info_dict['prc'] = float(cur_prc)
        info_dict.update(ef_dict)  # 添加 EF 指标
        return info_dict
    
    def _format_ef_key(self, pct):
        """
        生成稳定的 EF 键名
        0.005 -> "ef_0p5pct"
        0.01 -> "ef_1pct"
        0.02 -> "ef_2pct"
        """
        # 使用格式化保留1位小数，避免浮点精度问题
        pct_percent = pct * 100
        pct_str = f"{pct_percent:.1f}".replace('.', 'p')
        # 移除末尾的 p0（例如 1p0 -> 1）
        if pct_str.endswith('p0'):
            pct_str = pct_str[:-2]
        return f'ef_{pct_str}pct'
    
    def _compute_enrichment_factors(self, y_true, y_scores, top_pcts=None, min_samples=0):
        """
        计算多比例 Enrichment Factor 及相关指标
        
        Args:
            y_true: 真实标签 (0/1)
            y_scores: 预测分数（越高越可能是正例）
            top_pcts: 顶部比例列表，如 [0.005, 0.01, 0.02] 表示 0.5%, 1%, 2%
            min_samples: 计算 EF 所需的最小样本数
            
        Returns:
            dict: 各比例的 EF、TP、Precision、Recall、k 值
        """
        if top_pcts is None:
            top_pcts = [0.01]
        results = {}
        n_total = len(y_true)
        n_positives = int(y_true.sum())  # 总正例数
        
        # 总体正例率（baseline）
        overall_positive_rate = np.mean(y_true)
        if overall_positive_rate == 0:
            # 没有正例，无法计算 EF
            for pct in top_pcts:
                key_suffix = self._format_ef_key(pct).replace('ef_', '')
                results[f'k_{key_suffix}'] = 0
                results[f'ef_{key_suffix}'] = np.nan
                results[f'tp_{key_suffix}'] = 0
                results[f'prec_{key_suffix}'] = np.nan
                results[f'recall_{key_suffix}'] = np.nan
            return results
        
        # 按分数降序排序
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        for pct in top_pcts:
            k = max(1, int(np.ceil(n_total * pct)))  # 向上取整，至少取 1 个
            key_suffix = self._format_ef_key(pct).replace('ef_', '')
            
            if n_total < min_samples:
                # 样本不足，返回 nan
                results[f'k_{key_suffix}'] = k
                results[f'ef_{key_suffix}'] = np.nan
                results[f'tp_{key_suffix}'] = 0
                results[f'prec_{key_suffix}'] = np.nan
                results[f'recall_{key_suffix}'] = np.nan
                continue
            
            # 取前 k 个
            top_k_true = y_true_sorted[:k]
            tp = int(top_k_true.sum())  # TP@k
            precision_k = tp / k  # Precision@k
            recall_k = tp / (n_positives + 1e-12)  # Recall@k
            
            # EF = Precision@k / 总体正例率
            ef = precision_k / overall_positive_rate if overall_positive_rate > 0 else 0.0
            
            results[f'k_{key_suffix}'] = k
            results[f'ef_{key_suffix}'] = float(ef)
            results[f'tp_{key_suffix}'] = tp
            results[f'prec_{key_suffix}'] = float(precision_k)
            results[f'recall_{key_suffix}'] = float(recall_k)
        
        return results
    
    def _compute_enrichment_factors_scaffold_dedup(self, y_true, y_scores, scaffolds, top_pcts=None, max_per_scaffold=2, min_samples=0):
        """
        计算 Scaffold-dedup Enrichment Factor
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            scaffolds: 每个分子的 scaffold 列表（与 y_true/y_scores 对齐）
            top_pcts: 顶部比例列表
            max_per_scaffold: 每个 scaffold 最多取几个分子
            min_samples: 计算 EF 所需的最小样本数
            
        Returns:
            dict: 各比例的 scaffold-dedup EF 值
        """
        if top_pcts is None:
            top_pcts = [0.01]
        results = {}
        n_total = len(y_true)
        
        # 总体正例率
        overall_positive_rate = np.mean(y_true)
        if overall_positive_rate == 0:
            for pct in top_pcts:
                results[self._format_ef_key(pct) + '_scafdedup'] = 0.0
            return results
        
        if n_total < min_samples:
            for pct in top_pcts:
                results[self._format_ef_key(pct) + '_scafdedup'] = 0.0
            return results
        
        # 按分数降序排序，同时保留 scaffold 信息
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        scaffolds_sorted = [scaffolds[i] for i in sorted_indices]
        
        # 对每个 k 单独扫描排序列表，按 scaffold 限额选到 k 个为止
        for pct in top_pcts:
            k = max(1, int(np.ceil(n_total * pct)))  # 向上取整，至少取 1 个
            
            scaffold_counts = {}
            selected_labels = []
            
            # 扫描排序列表，直到凑够 k 个
            for label, scaffold in zip(y_true_sorted, scaffolds_sorted):
                count = scaffold_counts.get(scaffold, 0)
                if count < max_per_scaffold:
                    selected_labels.append(label)
                    scaffold_counts[scaffold] = count + 1
                    if len(selected_labels) >= k:
                        break
            
            if len(selected_labels) < k:
                # dedup 后样本不足
                results[self._format_ef_key(pct) + '_scafdedup'] = 0.0
                continue
            
            # 计算前 k 个的 EF
            top_k_true = selected_labels[:k]
            top_k_positive_rate = np.mean(top_k_true)
            
            ef = top_k_positive_rate / overall_positive_rate
            results[self._format_ef_key(pct) + '_scafdedup'] = float(ef)
        
        return results
    
    def _compute_per_protein_metrics(self, loader, min_samples_per_protein=10):
        """
        计算 Per-Protein 指标（用于 cold-protein 场景）
        
        Returns:
            dict: 包含 roc_macro, roc_micro, ef10_sufficient, precision10_sufficient
        """
        from collections import defaultdict
        from sklearn.metrics import roc_auc_score
        
        self.model.eval()
        
        # 收集所有预测和标签，按蛋白分组
        protein_data = defaultdict(lambda: {'y_true': [], 'y_scores': []})
        
        with torch.no_grad():
            for batch in loader:
                g, t, y = batch[:3]
                
                # 获取蛋白序列（从 batch 中提取）
                if len(batch) > 3 and isinstance(batch[3], dict) and 'protein' in batch[3]:
                    proteins = batch[3]['protein']
                else:
                    # 尝试从 t 中提取蛋白信息
                    proteins = ['unknown'] * len(y)
                
                g = g.to(self.device)
                y = y.to(self.device)
                
                t, mask, batch_use_precomputed = self._unpack_protein(t)
                
                model_output = self.model(g, t, mask=mask, use_precomputed_features=batch_use_precomputed)
                pred = self._extract_pred(model_output).sigmoid().view(-1).cpu().numpy()
                y_true_batch = y.view(-1).cpu().numpy()
                
                # 按蛋白分组
                for i, prot in enumerate(proteins):
                    if i < len(y_true_batch):
                        protein_data[prot]['y_true'].append(y_true_batch[i])
                        protein_data[prot]['y_scores'].append(pred[i])
        
        # 计算每个蛋白的指标
        roc_aucs = []
        ef10s = []
        precision10s = []
        
        for prot, data in protein_data.items():
            y_true = np.array(data['y_true'])
            y_scores = np.array(data['y_scores'])
            n_samples = len(y_true)
            
            if n_samples < 2:
                continue
            
            # 计算 ROC AUC（如果可能）
            if len(np.unique(y_true)) >= 2:
                try:
                    roc_auc = roc_auc_score(y_true, y_scores)
                    roc_aucs.append(roc_auc)
                except Exception:
                    pass
            
            # 计算 EF@10 和 Precision@10（如果样本足够）
            if n_samples >= min_samples_per_protein:
                k = min(10, n_samples)
                sorted_indices = np.argsort(y_scores)[::-1]
                top_k_indices = sorted_indices[:k]
                top_k_labels = y_true[top_k_indices]
                
                tp_at_k = np.sum(top_k_labels)
                precision_at_k = tp_at_k / k
                
                total_pos = np.sum(y_true)
                if total_pos > 0:
                    ef_at_k = (tp_at_k / k) / (total_pos / n_samples)
                else:
                    ef_at_k = 0.0
                
                ef10s.append(ef_at_k)
                precision10s.append(precision_at_k)
        
        # 计算宏平均
        results = {
            'roc_macro': np.mean(roc_aucs) if roc_aucs else 0.0,
            'roc_micro': 0.0,  # 暂不计算
            'ef10_sufficient': np.mean(ef10s) if ef10s else 0.0,
            'precision10_sufficient': np.mean(precision10s) if precision10s else 0.0,
            'n_proteins': len(protein_data),
            'n_proteins_sufficient': len(ef10s),
        }
        
        return results
    
    def select_threshold_on_valid(self, loader, criterion='f1', verbose=True):
        """
        在验证集上选择最优阈值
        
        Args:
            loader: 验证集数据加载器
            criterion: 阈值选择标准，可选 'f1' (F1-score) 或 'youden' (Youden J statistic)
            verbose: 是否打印详细信息
            
        Returns:
            dict: 包含最优阈值和相关信息
        """
        self.model.eval()
        y_true = []
        y_scores = []
        
        if self.show_progress and verbose:
            loader_iter = tqdm(loader, desc="selecting threshold on valid...", file=sys.stdout)
        else:
            loader_iter = loader
        
        for batch in loader_iter:
            g, t, y = batch[:3]
            
            g = g.to(self.device)
            y = y.to(self.device)
            
            t, mask, batch_use_precomputed = self._unpack_protein(t)
            
            with torch.no_grad():
                model_output = self.model(g, t, mask=mask, use_precomputed_features=batch_use_precomputed)
                pred = self._extract_pred(model_output).view(-1)
                y = y.view(-1).float()
                y_true.append(y)
                y_scores.append(pred.sigmoid())
        
        y_label = torch.cat(y_true, dim=0).cpu().numpy().reshape(-1)
        y_pred = torch.cat(y_scores, dim=0).cpu().numpy().reshape(-1)
        
        # 2.2 显式转 int，避免 sklearn 的奇怪行为
        y_label = y_label.astype(int)
        
        # 检查数据有效性
        if not np.isfinite(y_pred).all():
            print(f"⚠️  检测到非有限值在 valid y_pred 中！")
            return {
                'threshold': 0.5,
                'criterion': criterion,
                'criterion_value': 0.0,
                'positive_rate': 0.0,
                'threshold_f1': 0.5,
                'threshold_youden': 0.5,
                'f1_max': 0.0,
                'youden_j_max': 0.0,
            }
        
        if len(np.unique(y_label)) < 2:
            print(f"[WARNING] Valid set has only one class: {np.unique(y_label)}")
            return {
                'threshold': 0.5,
                'criterion': criterion,
                'criterion_value': 0.0,
                'positive_rate': float(np.mean(y_label)),
                'threshold_f1': 0.5,
                'threshold_youden': 0.5,
                'f1_max': 0.0,
                'youden_j_max': 0.0,
            }
        
        # 计算正例率（AUPRC baseline）
        positive_rate = np.mean(y_label)
        
        # 使用 ROC 曲线计算 Youden J（过滤掉非有限阈值）
        fpr, tpr, roc_thresholds = roc_curve(y_label, y_pred)
        # 过滤 inf 阈值，避免选出无意义阈值
        valid_mask = np.isfinite(roc_thresholds)
        fpr, tpr, roc_thresholds = fpr[valid_mask], tpr[valid_mask], roc_thresholds[valid_mask]
        youden_j = tpr - fpr
        
        # 使用 PR 曲线计算 F1
        prec, recall, pr_thresholds = precision_recall_curve(y_label, y_pred)
        precision_vals = prec[:-1]
        recall_vals = recall[:-1]
        f1_scores = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-8)
        
        # 计算 F1 最优阈值（使用 nanargmax 更稳，避免极端情况下出现 NaN）
        if len(f1_scores) > 0:
            best_idx_f1 = int(np.nanargmax(f1_scores))
            threshold_f1 = float(pr_thresholds[best_idx_f1]) if best_idx_f1 < len(pr_thresholds) else 0.5
            f1_max = float(f1_scores[best_idx_f1])
        else:
            threshold_f1 = 0.5
            f1_max = 0.0
        
        # 计算 Youden J 最优阈值（tie-breaking：选最接近 0.5 的阈值，避免极端值）
        if len(youden_j) > 0:
            max_j = np.max(youden_j)
            # 找到所有达到最大值的索引
            idxs = np.where(youden_j == max_j)[0]
            # tie-breaking：选最接近 0.5 的阈值
            best_idx_youden = idxs[np.argmin(np.abs(roc_thresholds[idxs] - 0.5))]
            threshold_youden = float(roc_thresholds[best_idx_youden])
            youden_j_max = float(youden_j[best_idx_youden])
        else:
            threshold_youden = 0.5
            youden_j_max = 0.0
        
        # 2.1 安全裁剪阈值到 [0, 1] 范围
        threshold_f1 = float(np.clip(threshold_f1, 0.0, 1.0))
        threshold_youden = float(np.clip(threshold_youden, 0.0, 1.0))
        
        # 根据请求的 criterion 返回对应的阈值
        if criterion == 'f1':
            best_threshold = threshold_f1
            best_criterion_value = f1_max
        elif criterion == 'youden':
            best_threshold = threshold_youden
            best_criterion_value = youden_j_max
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Use 'f1' or 'youden'.")
        
        if verbose:
            print(f"\n📊 Threshold Selection on Validation Set:")
            print(f"   F1-based threshold:     {threshold_f1:.4f} (F1={f1_max:.4f})")
            print(f"   Youden J-based threshold: {threshold_youden:.4f} (J={youden_j_max:.4f})")
            print(f"   Positive Rate (AUPRC baseline): {positive_rate:.4f}")
        
        return {
            'threshold': best_threshold,
            'criterion': criterion,
            'criterion_value': best_criterion_value,
            'positive_rate': positive_rate,
            'threshold_f1': threshold_f1,
            'threshold_youden': threshold_youden,
            'f1_max': f1_max,
            'youden_j_max': youden_j_max,
        }
    
    def test(self, loader, threshold=None):
        """
        Test the model with detailed debugging
        
        Args:
            loader: 测试集数据加载器
            threshold: 分类阈值，如果为 None 则使用 0.5
            
        Returns:
            list: [auroc, auprc, auprc_baseline, accuracy, sensitivity, specificity, precision, threshold_used]
        """
        try:
            self.model.eval()
            y_true = []
            y_scores = []
            loss_val = 0
            
            if getattr(self.args, 'debug_mask', False):
                print(f"[DEBUG] Starting test with loader length: {len(loader)}")
            if threshold is not None and getattr(self.args, 'debug_mask', False):
                print(f"[DEBUG] Using pre-selected threshold: {threshold:.4f}")
            
            if self.show_progress:
                loader_iter = tqdm(loader, desc="testing...", file=sys.stdout)
            else:
                loader_iter = loader
            
            for batch_idx, batch in enumerate(loader_iter):
                g, t, y = batch[:3]
                
                g = g.to(self.device)
                y = y.to(self.device)
                
                # 使用统一的 unpack 函数
                t, mask, batch_use_precomputed = self._unpack_protein(t)
                
                with torch.no_grad():
                    model_output = self.model(g, t, mask=mask, use_precomputed_features=batch_use_precomputed)
                    pred = self._extract_pred(model_output).view(-1)
                    y = y.view(-1).float()
                    loss = self.loss_function(pred, y)
                    loss_val += loss.item()
                    y_true.append(y)
                    y_scores.append(pred.sigmoid())
            
            if not y_true or not y_scores:
                print("WARNING: No data collected during test")
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, {}]
            
            loss_val /= len(loader)
            
            y_label = torch.cat(y_true, dim=0).cpu().numpy()
            y_pred = torch.cat(y_scores, dim=0).cpu().numpy()
            
            # ===== 🔴 额外保险：显式 reshape 成 1D =====
            y_label = y_label.reshape(-1)
            y_pred = y_pred.reshape(-1)
            
            # 2.2 显式转 int，避免 sklearn 的奇怪行为
            y_label = y_label.astype(int)
            
            # ===== 🔴 额外保险：提前检查 finite =====
            if not np.isfinite(y_pred).all():
                nan_count = np.sum(~np.isfinite(y_pred))
                print(f"⚠️  检测到 {nan_count} 个非有限值在 test y_pred 中！")
                raise ValueError("test y_pred 包含非有限值！")
            
            if getattr(self.args, 'debug_mask', False):
                print(f"[DEBUG] y_label shape={y_label.shape}, y_pred shape={y_pred.shape}")
            
            if len(y_label) == 0 or len(y_pred) == 0:
                print("WARNING: Empty data after concatenation")
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, {}]
            
            unique_labels = np.unique(y_label)
            if getattr(self.args, 'debug_mask', False):
                print(f"[DEBUG] Unique labels: {unique_labels}")
            
            # 计算正例率（AUPRC baseline）
            positive_rate = np.mean(y_label)
            
            if len(unique_labels) < 2:
                print(f"[WARNING] Only one class present: {unique_labels}")
                if 1 in unique_labels:
                    return [1.0, 1.0, positive_rate, 1.0, 1.0, 0.0, 1.0, threshold if threshold is not None else 0.5, {}]
                else:
                    return [1.0, 0.0, positive_rate, 1.0, 0.0, 1.0, 0.0, threshold if threshold is not None else 0.5, {}]
            
            try:
                auroc = roc_auc_score(y_label, y_pred)
            except Exception as e:
                print(f"[ERROR] Failed to compute AUC: {e}")
                auroc = 0.5
            
            try:
                auprc = average_precision_score(y_label, y_pred)
            except Exception as e:
                print(f"[ERROR] Failed to compute AUPRC: {e}")
                auprc = 0.0
            
            # 使用传入的阈值，如果没有则使用默认 0.5
            thred_optim = threshold if threshold is not None else 0.5
            
            # 使用 numpy 向量化（修复 2.4）
            y_pred_s = (y_pred >= thred_optim).astype(int)
            
            try:
                cm1 = confusion_matrix(y_label, y_pred_s, labels=[0, 1])
                
                if cm1.shape == (2, 2):
                    tn, fp, fn, tp = cm1.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                    if cm1.shape == (1, 1):
                        if cm1[0, 0] > 0:
                            if 1 in np.unique(y_label):
                                tp = cm1[0, 0]
                            else:
                                tn = cm1[0, 0]
            except Exception as e:
                print(f"[ERROR] Failed to compute confusion matrix: {e}")
                tn, fp, fn, tp = 0, 0, 0, 0
            
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / (total + 0.00001)
            sensitivity = tp / (tp + fn + 0.00001)
            specificity = tn / (tn + fp + 0.00001)
            precision1 = tp / (tp + fp + 0.00001)
            
            # 计算多比例 EF（用于虚拟筛选评估）
            ef_dict = self._compute_enrichment_factors(y_label, y_pred, [0.005, 0.01, 0.02])
            
            # 返回8个基础值 + EF 字典（兼容旧代码）
            results = [auroc, auprc, positive_rate, accuracy, sensitivity, specificity, precision1, thred_optim]
            
            # 将 EF 结果附加到返回列表的最后一个元素（字典形式）
            # 这样旧代码解包前8个值不受影响，新代码可以检查第8个元素是否为字典
            extended_results = results.copy()
            extended_results.append(ef_dict)  # 第9个元素：EF 字典
            
            return extended_results
        except Exception as e:
            print(f"ERROR in test: {e}")
            import traceback
            traceback.print_exc()
            # 修复：保证任何异常情况下都返回 9 个值（与正常路径一致）
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, threshold if threshold is not None else 0.5, {}]
    
    def test_with_scaffold_dedup(self, loader, threshold=None, smiles_col='SMILES', max_per_scaffold=1):
        """
        Test with scaffold-dedup EF calculation for virtual screening
        
        Args:
            loader: 测试集数据加载器（需要能访问 SMILES）
            threshold: 分类阈值
            smiles_col: SMILES 列名
            max_per_scaffold: 每个 scaffold 最多取几个分子
            
        Returns:
            dict: 包含所有评估指标，包括 scaffold-dedup EF
        """
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        try:
            self.model.eval()
            y_true = []
            y_scores = []
            smiles_list = []
            loss_val = 0
            
            if getattr(self.args, 'debug_mask', False):
                print(f"[DEBUG] Starting test_with_scaffold_dedup...")
            
            if self.show_progress:
                loader_iter = tqdm(loader, desc="testing with scaffold dedup...", file=sys.stdout)
            else:
                loader_iter = loader
            
            for batch_idx, batch in enumerate(loader_iter):
                g, t, y = batch[:3]
                
                # 尝试从 batch 获取 SMILES
                batch_smiles = None
                if len(batch) > 3:
                    if isinstance(batch[3], dict) and smiles_col in batch[3]:
                        batch_smiles = batch[3][smiles_col]
                    elif hasattr(batch[3], 'smiles'):
                        batch_smiles = batch[3].smiles
                
                # 如果 batch 中没有，尝试从 graph 中获取
                if batch_smiles is None and hasattr(g, 'smiles'):
                    if isinstance(g.smiles, list):
                        batch_smiles = g.smiles
                    else:
                        batch_smiles = [g.smiles] * len(y)
                
                if batch_smiles is None:
                    # 无法获取 SMILES，使用占位符
                    batch_smiles = [f"unknown_{batch_idx}_{i}" for i in range(len(y))]
                
                g = g.to(self.device)
                y = y.to(self.device)
                
                t, mask, batch_use_precomputed = self._unpack_protein(t)
                
                with torch.no_grad():
                    model_output = self.model(g, t, mask=mask, use_precomputed_features=batch_use_precomputed)
                    pred = self._extract_pred(model_output).view(-1)
                    y = y.view(-1).float()
                    loss = self.loss_function(pred, y)
                    loss_val += loss.item()
                    y_true.append(y)
                    y_scores.append(pred.sigmoid())
                    smiles_list.extend(batch_smiles if isinstance(batch_smiles, list) else [batch_smiles])
            
            if not y_true or not y_scores:
                print("WARNING: No data collected during test")
                return None
            
            loss_val /= len(loader)
            
            y_label = torch.cat(y_true, dim=0).cpu().numpy()
            y_pred = torch.cat(y_scores, dim=0).cpu().numpy()
            
            y_label = y_label.reshape(-1)
            y_pred = y_pred.reshape(-1)
            y_label = y_label.astype(int)
            
            if not np.isfinite(y_pred).all():
                nan_count = np.sum(~np.isfinite(y_pred))
                print(f"⚠️  检测到 {nan_count} 个非有限值在 test y_pred 中！")
                raise ValueError("test y_pred 包含非有限值！")
            
            # 计算基础指标
            positive_rate = np.mean(y_label)
            
            try:
                auroc = roc_auc_score(y_label, y_pred)
            except Exception as e:
                print(f"[ERROR] Failed to compute AUC: {e}")
                auroc = 0.5
            
            try:
                auprc = average_precision_score(y_label, y_pred)
            except Exception as e:
                print(f"[ERROR] Failed to compute AUPRC: {e}")
                auprc = 0.0
            
            # 计算阈值相关指标
            thred_optim = threshold if threshold is not None else 0.5
            y_pred_s = (y_pred >= thred_optim).astype(int)
            
            try:
                cm1 = confusion_matrix(y_label, y_pred_s, labels=[0, 1])
                if cm1.shape == (2, 2):
                    tn, fp, fn, tp = cm1.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
            except Exception:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / (total + 0.00001)
            sensitivity = tp / (tp + fn + 0.00001)
            specificity = tn / (tn + fp + 0.00001)
            precision1 = tp / (tp + fp + 0.00001)
            
            # 计算多比例 EF（普通）
            ef_dict = self._compute_enrichment_factors(y_label, y_pred, [0.005, 0.01, 0.02])
            
            # 计算 Murcko scaffolds
            def get_murcko_scaffold(smiles):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return f"INVALID::{smiles[:20]}"
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    return Chem.MolToSmiles(scaffold) if scaffold else smiles
                except Exception:
                    return smiles
            
            print(f"Computing Murcko scaffolds for {len(smiles_list)} molecules...")
            scaffolds = [get_murcko_scaffold(s) for s in smiles_list]
            
            # 计算 scaffold-dedup EF
            ef_dict_scaf = self._compute_enrichment_factors_scaffold_dedup(
                y_label, y_pred, scaffolds, 
                top_pcts=[0.005, 0.01, 0.02],
                max_per_scaffold=max_per_scaffold
            )
            
            # 打印结果对比
            print(f"\n📊 Virtual Screening Metrics:")
            print(f"  Total molecules: {len(y_label)}")
            print(f"  Unique scaffolds: {len(set(scaffolds))}")
            print(f"\n  Enrichment Factors (Standard):")
            for ef_name, ef_val in ef_dict.items():
                val_str = f"{ef_val:.2f}" if isinstance(ef_val, (int, float)) and np.isfinite(ef_val) else "nan"
                print(f"    {ef_name}: {val_str}")
            print(f"\n  Enrichment Factors (Scaffold-dedup, max_per_scaffold={max_per_scaffold}):")
            for ef_name, ef_val in ef_dict_scaf.items():
                val_str = f"{ef_val:.2f}" if isinstance(ef_val, (int, float)) and np.isfinite(ef_val) else "nan"
                print(f"    {ef_name}: {val_str}")
            
            return {
                'auroc': auroc,
                'auprc': auprc,
                'positive_rate': positive_rate,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision1,
                'threshold': thred_optim,
                'ef': ef_dict,
                'ef_scaffold_dedup': ef_dict_scaf,
                'scaffold_dedup_config': {
                    'method': 'MurckoScaffold',
                    'max_per_scaffold': max_per_scaffold,
                },
                'n_molecules': len(y_label),
                'n_scaffolds': len(set(scaffolds)),
            }
            
        except Exception as e:
            print(f"ERROR in test_with_scaffold_dedup: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_info(self, info_dict, desc="Validation info: "):
        info = ""
        for key, value in info_dict.items():
            if isinstance(value, dict):
                info += f'{key}: '
                for k, v in value.items():
                    # ✅ 加 isinstance 保护，避免非数值类型触发 TypeError
                    if isinstance(v, (int, float, np.floating)):
                        info += f'{k}: {v:.4f}  ' if np.isfinite(v) else f'{k}: nan  '
                info += '\t'
            elif isinstance(value, (int, float, np.floating)):
                info += f'{key}: {value:.4f}\t' if np.isfinite(value) else f'{key}: nan\t'
        print(desc, end='') 
        print(info)
        return info

    def log_info(self, info_dict, desc="Validation info: "):
        info = self.print_info(info_dict, desc)
        info += "\n"
        with open(f"log/DTI/detail/{self.run_time}.txt", "a") as f:
            f.write(desc + info)
    
    def check_pool(self):
        smiles = 'C1=CC=C(C=C1)C2=CC(=O)C3=C(C(=C(C=C3O2)O[C@H]4[C@@H]([C@H]([C@@H]([C@H](O4)C(=O)O)O)O)O)O)O'
        data = from_smiles(smiles)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        data = data.to(self.device)
        edge_index = data.edge_index
        self.model.eval()
        with torch.no_grad():
            _ = self.model(data)
        
        comps, tars = self.model.unet.info()
        mol = mol_with_atom_index(Chem.MolFromSmiles(smiles))
        imgs = comps_visualize_multi(mol, comps, tars, edge_index)
        for png in imgs:
            display(Image(png))
