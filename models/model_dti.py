# -*- coding: utf-8 -*-
"""
model_dti.py
Created on Tue Mar 14 15:12:06 2023

@author: Fanding Xu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, to_dense_batch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.models import MLP
from torch.nn.utils.weight_norm import weight_norm
from .MolUnet import MolUnetEncoder
from .layers import FPNN
from transformers import EsmModel

class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        # 使用 padding=k//2 实现 same padding（兼容老版本PyTorch）
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0], padding=kernels[0]//2)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding=kernels[1]//2)
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding=kernels[2]//2)
        self.bn3 = nn.BatchNorm1d(in_ch[3])
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, v, mask=None, return_pooled=False):
        """
        Args:
            v: [B, L] - protein token sequence
            mask: [B, L] - attention mask (1=valid, 0=pad)
            return_pooled: if True, return pooled features [B, C]; else return residue features [B, L, C]
        
        Returns:
            if return_pooled: pooled [B, C]
            else: residue_feat [B, L, C]
        
        Note: Conv1d with padding=k//2 preserves sequence length L,
              so mask can be directly applied to output features.
        """
        v = self.embedding(v.long())  # [B, L, C]
        v = v.transpose(2, 1)  # [B, C, L]
        v = self.bn1(F.relu(self.conv1(v)))  # [B, C', L] (L preserved)
        v = self.bn2(F.relu(self.conv2(v)))  # [B, C', L] (L preserved)
        v = self.bn3(F.relu(self.conv3(v)))  # [B, C', L] (L preserved)
        
        if return_pooled:
            if mask is not None:
                m = mask.unsqueeze(1).float()  # [B, 1, L]
                denom = m.sum(dim=2).clamp(min=1.0)  # [B, 1]
                avg_pooled = (v * m).sum(dim=2) / denom  # [B, C']
                
                neg = -1e4 if v.dtype in (torch.float16, torch.bfloat16) else -1e9
                v_for_max = v.masked_fill(m == 0, neg)
                max_pooled = v_for_max.max(dim=2).values  # [B, C']
            else:
                max_pooled = self.global_max_pool(v).squeeze(-1)
                avg_pooled = self.global_avg_pool(v).squeeze(-1)
            
            pooled = (max_pooled + avg_pooled) / 2  # [B, C']
            return pooled
        else:
            v = v.transpose(1, 2)  # [B, L, C']
            return v


class ProteinESM(nn.Module):
    def __init__(self,
                 embedding_dim: int = None,          # 希望最终拿到多少维，None表示自动根据模型大小设置
                 pretrained_model: str = './dataset/data/esm2_t30_150M_UR50D/',
                 finetune: str = 'False'):  # 修改为支持 'False', 'True', 'partial'
        super().__init__()
        import os
        import torch
        from transformers import EsmModel, EsmConfig
        
        # 直接从本地文件加载模型，完全避开from_pretrained的路径验证
        model_dir = pretrained_model
        
        # 加载配置
        config_path = os.path.join(model_dir, 'config.json')
        config = EsmConfig.from_json_file(config_path)
        
        # 创建模型实例
        self.esm = EsmModel(config)
        
        # 加载模型权重（支持两种格式）
        weights_path = os.path.join(model_dir, 'pytorch_model.bin')
        if not os.path.exists(weights_path):
            # 尝试safetensors格式
            weights_path = os.path.join(model_dir, 'model.safetensors')
            if os.path.exists(weights_path):
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
            else:
                raise FileNotFoundError(f"未找到模型权重文件: {os.path.join(model_dir, 'pytorch_model.bin')} 或 {os.path.join(model_dir, 'model.safetensors')}")
        else:
            state_dict = torch.load(weights_path, map_location='cpu')
        
        # 处理权重字典，移除"esm."前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('esm.'):
                # 移除"esm."前缀
                new_key = key[4:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 加载权重，使用strict=False允许忽略不匹配的键
        # 这是必要的，因为下载的权重文件包含了语言模型头(lm_head)的权重，而我们不需要这些
        self.esm.load_state_dict(new_state_dict, strict=False)
        
        # 开启梯度检查点，减少显存使用
        # 注意：只在训练模式下启用，避免影响特征计算的确定性
        if finetune != 'False':
            self.esm.gradient_checkpointing_enable()
            print(f"✅ gradient_checkpoint: {self.esm.is_gradient_checkpointing}")
        else:
            # 在冻结模式下禁用梯度检查点，确保特征计算的一致性
            self.esm.gradient_checkpointing_disable()
            print(f"✅ gradient_checkpoint disabled for frozen mode")
        
        self.esm_feature_dim = self.esm.config.hidden_size   # 1280 for 650M
        
        # 根据模型大小自动设置合适的embedding_dim
        if embedding_dim is None:
            # 获取模型名称或路径中的关键信息
            model_info = pretrained_model.lower()
            hidden_size = self.esm_feature_dim
            
            if 't12' in model_info or '35m' in model_info or hidden_size == 480:
                # 35M (t12): hidden_size = 480
                self.proj_dim = 128
            elif 't30' in model_info or '150m' in model_info or hidden_size == 640:
                # 150M (t30): hidden_size = 640
                self.proj_dim = 256
            elif 't33' in model_info or '650m' in model_info or hidden_size == 1280:
                # 650M: hidden_size = 1280
                self.proj_dim = 512
            else:
                # 默认值
                self.proj_dim = 128
            print(f"🔍 Auto-set proj_dim: {self.proj_dim} (based on model size: {hidden_size})")
        else:
            self.proj_dim = embedding_dim
            print(f"🔍 User-specified proj_dim: {self.proj_dim}")
        
        self.proj = nn.Linear(self.esm_feature_dim, self.proj_dim)

        # 冻结模型参数
        if finetune == 'False':
            for p in self.esm.parameters():
                p.requires_grad = False
        elif finetune == 'partial':
            # 只冻结除最后2层transformer外的参数
            for name, param in self.esm.named_parameters():
                # 解冻最后2层transformer
                if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        # 如果finetune == 'True'，则所有参数都可训练
        
        # 打印可训练参数
        print("\n🔍 Trainable parameters:")
        trainable_count = 0
        for name, param in self.esm.named_parameters():
            if param.requires_grad:
                print(f"  - {name}")
                trainable_count += 1
        print(f"\n📊 Total trainable parameters: {trainable_count}")
        print(f"📊 ESM model size: {sum(p.numel() for p in self.esm.parameters()):,} parameters")

    def forward(self, tokens, attention_mask=None):
        """
        tokens:  LongTensor  [B, L]  (已经 pad 好的 token id)
        attention_mask: LongTensor [B, L] (可选，1=有效，0=pad)
        return:  FloatTensor [B, L, proj_dim]
        """
        if attention_mask is not None:
            out = self.esm(input_ids=tokens, attention_mask=attention_mask).last_hidden_state
        else:
            out = self.esm(tokens).last_hidden_state
        out = self.proj(out)                            # [B, L, proj_dim]
        # 替换NaN值
        out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        return out


class ProteinESMCNN(nn.Module):
    """
    对ESM预计算的残基级特征应用CNN
    输入: [B, L, esm_dim] (预计算的ESM特征)
    输出: [B, output_dim] (池化后的全局特征)
    """
    def __init__(self, esm_dim=256, hidden_dims=[256, 128], 
                 kernel_sizes=[7, 5], output_dim=128, dropout=0.2):
        super().__init__()
        
        # 构建卷积层
        layers = []
        in_dim = esm_dim
        
        for hidden_dim, k_size in zip(hidden_dims, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=k_size, 
                         padding=k_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 双池化策略 (max + mean)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 最终投影 (拼接max和mean pooling)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 用于BAN的残基级投影 (来自CNN卷积后的特征，与query同源)
        self.residue_proj = nn.Linear(hidden_dims[-1], output_dim)
        
        print(f"✅ ProteinESMCNN initialized:")
        print(f"   - Input dim: {esm_dim}")
        print(f"   - Hidden dims: {hidden_dims}")
        print(f"   - Kernel sizes: {kernel_sizes}")
        print(f"   - Output dim: {output_dim}")
    
    def forward(self, x, mask=None, return_residue_features=False):
        """
        x: [B, L, esm_dim] - ESM预计算的残基级特征
        mask: [B, L] - attention mask (可选，1=有效，0=pad)
        return_residue_features: 是否返回残基级特征(用于BAN)
        
        Returns:
            - pooled: [B, output_dim] - 池化后的全局特征
            - residue_feat: [B, L, output_dim] - 残基级特征(可选)
        """
        # 输入检查和 NaN 处理
        if torch.isnan(x).any():
            print("⚠️ Warning: NaN detected in ProteinESMCNN input")
            x = torch.nan_to_num(x, nan=0.0)
        
        # 转换为Conv1d格式: [B, esm_dim, L]
        x_conv = x.transpose(1, 2)  # [B, esm_dim, L]
        
        # 卷积层
        x_conv = self.conv_layers(x_conv)  # [B, hidden_dim, L]
        
        # 检查卷积后是否有NaN
        if torch.isnan(x_conv).any():
            print("⚠️ Warning: NaN detected after conv layers")
            x_conv = torch.nan_to_num(x_conv, nan=0.0)
        
        # 双池化 (支持 masked pooling)
        if mask is not None:
            m = mask.unsqueeze(1).float()  # [B, 1, L]
            
            # masked avg
            denom = m.sum(dim=2).clamp(min=1.0)  # [B, 1]
            avg_pooled = (x_conv * m).sum(dim=2) / denom  # [B, hidden_dim]
            
            # masked max: 把 pad 位置设为有限大负数（避免 fp16 溢出）
            neg = -1e4 if x_conv.dtype in (torch.float16, torch.bfloat16) else -1e9
            x_for_max = x_conv.masked_fill(m == 0, neg)
            max_pooled = x_for_max.max(dim=2).values  # [B, hidden_dim]
        else:
            max_pooled = self.global_max_pool(x_conv).squeeze(-1)  # [B, hidden_dim]
            avg_pooled = self.global_avg_pool(x_conv).squeeze(-1)  # [B, hidden_dim]
        
        # 拼接并投影
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # [B, hidden_dim*2]
        pooled = self.fc(pooled)  # [B, output_dim]
        
        if return_residue_features:
            # 为BAN提供投影后的残基级特征 (来自CNN卷积后，与query同源)
            residue_feat = self.residue_proj(x_conv.transpose(1, 2))  # [B, L, output_dim]
            # 检查 residue_feat 的 NaN
            if torch.isnan(residue_feat).any():
                print("⚠️ Warning: NaN detected in residue_feat")
                residue_feat = torch.nan_to_num(residue_feat, nan=0.0)
            return pooled, residue_feat
        else:
            return pooled


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
    
class ScoreJK(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.nn = nn.Sequential(nn.Linear(in_dim, in_dim),
                                nn.ReLU(),
                                nn.Linear(in_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.nn[0].weight)
        self.nn[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.nn[-1].weight)
        self.nn[-1].bias.data.zero_()

    def forward(self, src):
        src = torch.stack(src, dim=1)
        a = self.nn(src)
        a = torch.softmax(a, dim=1)
        out = (a * src).sum(dim=1)
        return out
     
class UnetDTI(nn.Module):
    @staticmethod
    def _infer_esm_dim(model_name):
        """根据 ESM 模型名推断特征维度"""
        model_lower = model_name.lower()
        if 't30' in model_lower or '150m' in model_lower:
            return 256
        elif 't33' in model_lower or '650m' in model_lower:
            return 512
        elif 't12' in model_lower or '35m' in model_lower:
            return 128
        else:
            return 128
    
    def __init__(self, config, protein_extractor_type='cnn', esm_model='esm2_t30_150M_UR50D', finetune='partial', use_precomputed_features=False):
        super().__init__()
        self.esm_model_name = esm_model  # 保存模型名称（字符串）
        self.esm_encoder = None  # ESM 编码器模块（初始为 None）
        self.protein_extractor_type = protein_extractor_type  # 保存类型用于 forward
        # 预计算特征支持
        self.use_precomputed_features = use_precomputed_features
        
        # =========================================
        #  🔑 第一步：强制固定 hidden_dim = 128（对比实验）
        # =========================================
        config['model']['hidden_dim'] = 128
        config['FP']['hidden_dims'] = [128, 128, 128]  # 确保 FP 层也一致
        
        # 读取固定后的值
        out_dim = config['model']['hidden_dim']  # 128
        num_pool_layer = config['model']['num_pool_layer']
        
        # 先设置这些属性，后面会用到
        self.jk = config['predict']['jk'] = "cat"
        self.pool_first = config['predict']['pool_first']
        self.graph_pool = config['predict']['graph_pool']
        self.dropout_rate = config['predict']['dropout_rate']
        
        print(f"🔍 Fixed configuration for fair comparison:")
        print(f"   - hidden_dim: {out_dim}")
        print(f"   - num_pool_layer: {num_pool_layer}")
        
        # =========================================
        #           DrugBAN settings
        # =========================================
        protein_emb_dim = 128
        num_filters = [128, 128, 128]
        kernel_size = [3, 7, 9]  # 改为全奇数，兼容 padding=k//2
        protein_padding = True
        ban_heads = 2
        mlp_in_dim = 256
        mlp_hidden_dim = 512
        mlp_out_dim = 128
        out_binary = 1
        
        self.query_fp = config['FP']['query_fp'] = True
        # config['FP']['hidden_dims'] 已经在上面设为 [128,128,128]
        
        import os
        
        # 存储蛋白质特征维度
        self.protein_feature_dim = num_filters[-1]  # 默认值
        
        # 选择蛋白质提取器类型
        if protein_extractor_type == 'cnn':
            self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
            self.protein_feature_dim = num_filters[-1]  # 显式设置，防未来修改踩坑
        elif protein_extractor_type == 'esm':
            if use_precomputed_features:
                self.protein_extractor = None  # 不需要加载ESM模型
                # ========== 从 esm_model 名称推断维度 ==========
                self.protein_feature_dim = self._infer_esm_dim(esm_model)
                
                print("✅ Using precomputed protein features, skipping ESM model loading")
                print(f"🔍 Inferred protein_feature_dim: {self.protein_feature_dim}")
            else:
                # 使用ESM-2模型
                pretrained_model = f'./dataset/data/{self.esm_model_name}/'
                # 不指定embedding_dim，让ProteinESM自动设置
                self.protein_extractor = ProteinESM(
                    embedding_dim=None,  # 自动根据模型大小设置
                    pretrained_model=pretrained_model, 
                    finetune=finetune
                )
                # 更新蛋白质特征维度
                self.protein_feature_dim = self.protein_extractor.proj_dim
                print(f"🔍 Updated protein_feature_dim: {self.protein_feature_dim}")
        elif protein_extractor_type == 'esm_cnn':
            if use_precomputed_features:
                # 从模型名推断ESM特征维度
                esm_dim = self._infer_esm_dim(self.esm_model_name)
                
                # 初始化ESM+CNN编码器（只用CNN）
                self.protein_extractor = ProteinESMCNN(
                    esm_dim=esm_dim,
                    hidden_dims=[256, 128],  # 可调整
                    kernel_sizes=[7, 5],      # 可调整
                    output_dim=128,           # 统一到128
                    dropout=0.2
                )
                self.protein_feature_dim = 128  # CNN输出维度
                print(f"✅ Using ESM+CNN mode (precomputed, esm_dim={esm_dim} → output=128)")
            else:
                # 即时计算模式：先加载ESM模型，再用CNN
                pretrained_model = f'./dataset/data/{self.esm_model_name}/'
                self.esm_encoder = ProteinESM(
                    embedding_dim=None,  # 自动根据模型大小设置
                    pretrained_model=pretrained_model, 
                    finetune=finetune
                )
                # 从ESM模型获取特征维度
                esm_dim = self.esm_encoder.proj_dim
                # 初始化ESM+CNN编码器
                self.protein_extractor = ProteinESMCNN(
                    esm_dim=esm_dim,
                    hidden_dims=[256, 128],  # 可调整
                    kernel_sizes=[7, 5],      # 可调整
                    output_dim=128,           # 统一到128
                    dropout=0.2
                )
                self.protein_feature_dim = 128  # CNN输出维度
                print(f"✅ Using ESM+CNN mode (on-the-fly, esm_dim={esm_dim} → output=128)")
        else:
            raise ValueError(f"Unknown protein extractor type: {protein_extractor_type}")
        
        # ========== 计算正确的分子特征维度（现在安全了）==========
        # 根据 JK 类型确定实际的分子特征维度
        if self.jk == "cat" and not self.pool_first:
            self.mol_feature_dim = num_pool_layer * out_dim  # 3 * 128 = 384
        else:
            self.mol_feature_dim = out_dim  # 128
        
        print(f"🔍 Molecular feature dim: {self.mol_feature_dim} (JK={self.jk}, pool_first={self.pool_first})")
        
        # ========== 添加 protein 到 query 的投影层（专业修复）==========
        # 只加 pre-norm，避免双层 norm 让表示太“白”
        self.prot_norm = nn.LayerNorm(self.protein_feature_dim)
        self.protein_to_query = nn.Linear(self.protein_feature_dim, out_dim)
        # 🔑 更保守的初始化：用更小的标准差
        torch.nn.init.normal_(self.protein_to_query.weight, mean=0.0, std=0.01)
        if self.protein_to_query.bias is not None:
            torch.nn.init.zeros_(self.protein_to_query.bias)
        print(f"🔍 Added protein_to_query projection (with pre-norm): {self.protein_feature_dim} -> {out_dim}")
        
        # 使用正确的维度初始化 BAN 层
        self.bcn = weight_norm(
            BANLayer(v_dim=self.mol_feature_dim, 
                     q_dim=self.protein_feature_dim,  # 256 (150M) 或 128 (35M)
                     h_dim=mlp_in_dim, 
                     h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        # ========================================= 
        config['MP']['norm'] = "BatchNorm"
        config['MP']['act'] = "ReLU"
        config['MP']['heads'] = 2
        config['FP']['norm'] = "BatchNorm"
        config['FP']['act'] = "ReLU"
        
        out_dim = config['model']['hidden_dim']
        num_pool_layer = config['model']['num_pool_layer']
        graph_pool = config['predict']['graph_pool']
        self.graph_pool = graph_pool
        
        feature_dim = num_pool_layer * out_dim if self.jk == "cat" and not self.pool_first else out_dim
        self.unet = MolUnetEncoder(config)
        
        
            
        feature_dim = num_pool_layer * out_dim if self.jk == "cat" and not self.pool_first else out_dim
        if self.jk == "score":
            self.score_jk = ScoreJK(out_dim)
        
        #Different kind of graph pooling
        if graph_pool == "sum":
            self.pool = global_add_pool
        elif graph_pool == "mean":
            self.pool = global_mean_pool
        elif graph_pool == "max":
            self.pool = global_max_pool
        elif graph_pool == "attention":
            if self.pool_first:
                self.pool = nn.Sequential(
                    *[GlobalAttention(gate_nn = torch.nn.Linear(feature_dim, 1)) for i in range(num_pool_layer)])
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(feature_dim, 1))
                
        elif graph_pool[:-1] == "set2set":
            set2set_iter = int(graph_pool[-1])
            if self.pool_first:
                self.pool = nn.Sequential(
                    *[Set2Set(feature_dim, set2set_iter) for i in range(num_pool_layer)])
            else:
                self.pool = Set2Set(feature_dim, set2set_iter)
            feature_dim = feature_dim * 2
        else:
            raise ValueError("Invalid graph pooling type.")
        
        if self.pool_first:
            feature_dim *= num_pool_layer
        
        # self.attr_decoder_mol = MLP([feature_dim, out_dim])
        # self.attr_decoder_mol = MLP([feature_dim, feature_dim*2, out_dim])
        # self.attr_decoder_pro = MLP([128, 256, out_dim])
        
        # final_dim = out_dim
        # self.predict = nn.Sequential(nn.Linear(final_dim, final_dim),
        #                              nn.BatchNorm1d(final_dim),
        #                              nn.ReLU(),
        #                              nn.Linear(final_dim, 1))
        
        self.gnn = DrugGCN(out_dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        pass

    
    def forward(self, g, t, mask=None, use_precomputed_features=None):
        """Forward pass with ESM+CNN support"""
        
        def check_nan(tensor, name):
            if tensor is not None and torch.isnan(tensor).any():
                print(f"NaN detected in {name}")
                return True
            return False
        
        # Check inputs
        has_nan = check_nan(g.x, "g.x")
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            has_nan |= check_nan(g.edge_attr, "g.edge_attr")
        has_nan |= check_nan(t, "t")
        
        # ===== 🔑 语义清晰：调用者显式传参覆盖默认行为 =====
        use_precomp = self.use_precomputed_features if use_precomputed_features is None else use_precomputed_features
        
        # ===== 🛡️ 防走错保护：预计算模式必须有 mask =====
        if use_precomp and mask is None:
            raise ValueError(
                "Precomputed protein features require attention mask (built from true sequence length). "
                "Please provide mask in dataset/collate or set use_precomputed_features=False."
            )
        
        # ===== 🔑 统一 mask 为 batched 版本（避免1D/2D混乱）=====
        mask_b = mask
        if mask_b is not None and mask_b.dim() == 1:
            mask_b = mask_b.unsqueeze(0)  # [1, L]
        
        # ===== 规范 mask_b 的 dtype/取值（0/1 long）=====
        if mask_b is not None:
            mask_b = (mask_b > 0).to(dtype=torch.long)
        
        # ===== 初始化 orig_tokens，避免未定义 =====
        orig_tokens = None
        
        # ===== 处理蛋白质输入 =====
        if not isinstance(t, torch.Tensor):
            try:
                t = torch.tensor(t, dtype=torch.float32 if use_precomp else torch.long)
            except Exception as e:
                raise ValueError(f"Failed to convert t to tensor: {e}")
        
        # ===== 🆕 关键: esm_cnn 即时计算模式 =====
        if self.protein_extractor_type == 'esm_cnn' and not use_precomp and isinstance(self.esm_encoder, ProteinESM):
            # 即时计算模式：先通过 ESM 提取特征
            if t.dim() == 2 and t.dtype == torch.long:
                # [B, L] tokens
                protein_feat = self.esm_encoder(t, attention_mask=mask_b)  # [B, L, esm_dim]
            elif t.dim() == 1 and t.dtype == torch.long:
                # [L] tokens
                protein_feat = self.esm_encoder(t.unsqueeze(0), attention_mask=mask_b)  # [1, L, esm_dim]
            else:
                raise ValueError(f"esm_cnn on-the-fly mode expects tokens, got shape={t.shape}, dtype={t.dtype}")
        else:
            # 原始逻辑
            # 确保是3D张量 [B, L, D]
            if t.dim() == 3:
                # Already features [B, L, D], use directly
                protein_feat = t
            elif t.dim() == 2:
                # 关键：根据数据类型区分特征和 tokens
                if t.dtype in [torch.float32, torch.float64]:
                    # 2D float 特征是错误的，预计算特征必须是 3D [B, L, D]
                    raise ValueError(
                        f"Precomputed features must be 3D [B, L, D], got 2D: shape={t.shape}. "
                        f"Please check your data loader or feature precomputation."
                    )
                else:
                    # 整数，是 tokens [B, L] - 保存原始 tokens 用于后续 pool
                    orig_tokens = t
                    if use_precomp:
                        raise ValueError(f"use_precomputed_features=True but got integer tokens: shape={t.shape}, dtype={t.dtype}")
                    if self.protein_extractor is None:
                        raise ValueError("protein_extractor is None, but use_precomputed_features is False")
                    # 若 extractor 支持 attention_mask，就传进去
                    if isinstance(self.protein_extractor, ProteinESM):
                        protein_feat = self.protein_extractor(t, attention_mask=mask_b)
                    elif isinstance(self.protein_extractor, ProteinCNN):
                        protein_feat = self.protein_extractor(t, mask=mask_b, return_pooled=False)
                    else:
                        protein_feat = self.protein_extractor(t)
            elif t.dim() == 1:
                if use_precomp:
                    # 预计算模式下 1D 输入是错误的
                    raise ValueError(f"use_precomputed_features=True but got 1D input: shape={t.shape}, dtype={t.dtype}")
                else:
                    # [L] tokens，添加 batch 维度后提取特征 - 保存原始 tokens 用于后续 pool
                    orig_tokens = t.unsqueeze(0)
                    if self.protein_extractor is None:
                        raise ValueError("protein_extractor is None, but use_precomputed_features is False")
                    # 若 extractor 支持 attention_mask，就传进去
                    am = mask_b
                    if isinstance(self.protein_extractor, ProteinESM):
                        protein_feat = self.protein_extractor(t.unsqueeze(0), attention_mask=am)
                    elif isinstance(self.protein_extractor, ProteinCNN):
                        protein_feat = self.protein_extractor(t.unsqueeze(0), mask=am, return_pooled=False)
                    else:
                        protein_feat = self.protein_extractor(t.unsqueeze(0))
            else:
                # Unexpected dimension
                raise ValueError(f"Unexpected t dimension: {t.dim()}, shape: {t.shape}, dtype: {t.dtype}")
        
        # ===== 🔑 CRITICAL: Final dimension check =====
        assert protein_feat.dim() == 3, f"protein features must be 3D tensor, got {protein_feat.dim()}D: {protein_feat.shape}"
        t = protein_feat
        has_nan |= check_nan(t, "protein features after processing")
        
        # ===== 🆕 关键修改: 根据编码器类型处理特征 =====
        if isinstance(self.protein_extractor, ProteinESMCNN):
            # ESM+CNN模式: 同时获取池化特征和残基级特征
            query, residue_feat = self.protein_extractor(
                t, mask=mask_b, return_residue_features=True
            )
            # query: [B, 128] 用于unet (已做 masked pooling)
            # residue_feat: [B, L, 128] 用于BAN
        elif isinstance(self.protein_extractor, ProteinCNN):
            # CNN模式: 使用 masked max+avg pooling（与ESM-CNN保持一致）
            if orig_tokens is None:
                raise ValueError("ProteinCNN requires token inputs; got precomputed features.")
            pooled = self.protein_extractor(orig_tokens, mask=mask_b, return_pooled=True)
            pooled = self.prot_norm(pooled)
            query = self.protein_to_query(pooled)
            residue_feat = t  # [B, L, D] 原始残基特征
        else:
            # 原始模式: mean pooling (支持 masked mean)
            if mask_b is not None and mask_b.dim() == 2:
                # 有 attention_mask，进行 masked mean pooling
                mask_expanded = mask_b.unsqueeze(-1).float()  # [B, L, 1]
                denom = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
                protein_mean = (t * mask_expanded).sum(dim=1) / denom  # [B, D]
            else:
                # 无 mask，使用普通 mean pooling
                protein_mean = t.mean(dim=1)  # [B, D]
            
            protein_mean = self.prot_norm(protein_mean)
            query = self.protein_to_query(protein_mean)  # [B, 128]
            residue_feat = t  # [B, L, D] 原始残基特征
        
        has_nan |= check_nan(query, "query after processing")
        
        # ===== 用 attention_mask 把 residue_feat 的 pad 位置清零 =====
        if mask_b is not None and mask_b.dim() == 2:
            mask_expanded = mask_b.unsqueeze(-1).float()  # [B, L, 1]
            residue_feat = residue_feat * mask_expanded  # 清零 pad 位置
        
        # ===== 处理分子图 =====
        batch = g.batch
        xs, _, _ = self.unet(g, query=query)
        x = self.__do_jk(xs)
        has_nan |= check_nan(x, "drug features after extraction")
        
        # ===== 🔑 断言：确认 x 是 2D 节点特征 =====
        assert x.dim() == 2, f"Expected node features [num_nodes, dim], got {x.shape}"
        
        # ===== BAN预测 =====
        dx, dm = to_dense_batch(x, batch)
        # dm 已经是 bool mask (True=有效节点)，直接用
        att, logits = self.BAN_pred(dx, residue_feat, q_mask=mask_b, v_mask=dm)
        has_nan |= check_nan(logits, "predictions after BAN")
        
        return logits, (x, att)

    def BAN_pred(self, dx, t, q_mask=None, v_mask=None):
        """BAN prediction with dimension checks and NaN checking"""
        def check_nan(tensor, name):
            if tensor is not None and torch.isnan(tensor).any():
                print(f"NaN detected in BAN_pred: {name}")
                return True
            return False
        
        # ===== 🔑 CRITICAL: Check dimensions =====
        assert t.dim() == 3, f"t must be 3D before BAN_pred, got {t.dim()}D: {t.shape}"
        assert dx.dim() == 3, f"dx must be 3D before BAN_pred, got {dx.dim()}D: {dx.shape}"
        
        # Check NaN inputs
        check_nan(dx, "dx input")
        check_nan(t, "t input")
        
        # ===== 🔑 CRITICAL: Check mask length =====
        if q_mask is not None:
            if q_mask.dim() != 2:
                raise ValueError(f"q_mask must be [B, L], got {q_mask.shape}")
            if q_mask.size(1) != t.size(1):
                raise ValueError(
                    f"q_mask length mismatch: q_mask.size(1)={q_mask.size(1)} vs t.size(1)={t.size(1)}. "
                    f"Did you forget to pad CNN to same length?"
                )
        
        if v_mask is not None:
            if v_mask.dim() != 2:
                raise ValueError(f"v_mask must be [B, V], got {v_mask.shape}")
            if v_mask.size(1) != dx.size(1):
                raise ValueError(
                    f"v_mask length mismatch: v_mask.size(1)={v_mask.size(1)} vs dx.size(1)={dx.size(1)}"
                )
        
        # BAN interaction
        f, att = self.bcn(dx, t, q_mask=q_mask, v_mask=v_mask, softmax=True)
        check_nan(f, "after ban attention")
        check_nan(att, "after ban logits")
        
        # Classifier
        logits = self.mlp_classifier(f)
        logits = logits.squeeze(-1)
        check_nan(logits, "after mlp classifier")
        
        return att, logits

    
    def __do_jk(self, src):
        if self.jk == "cat":
            x = torch.cat(src, dim = 1)
        elif self.jk == "last":
            x = src[-1]
        elif self.jk == "max":
            x = torch.max(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "mean":
            x = torch.mean(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "sum" or self.jk == "add":
            x = torch.sum(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "score":
            x = self.score_jk(src)
        else:
            raise ValueError("Invalid JK type.")
        return x
        
    def cal_attention(self, g):
        batch = g.batch
        xs, es = self.unet(g)
        if self.pool_first:
            attn = [pool.gate_nn(x) for pool, x in zip(self.pool, xs)]
            attn = [softmax(a, batch) for a in attn]
        else:
            x = self.__do_jk(xs)
            attn = self.pool.gate_nn(x)
            attn = softmax(attn, batch)
        return attn.view(-1)
    
    @property
    def pool_info(self):
        return self.unet.pool_info



# ============================== DrugBAN =====================================
from torch_geometric.nn import GCN, GCNConv

class UNetBAN(MolUnetEncoder):
    def __init__(self, config):
        super().__init__(config)
        out_dim = config['model']['hidden_dim']
        self.atom_embedding = nn.Linear(74, out_dim)
        self.bond_embedding = nn.Linear(12, out_dim)
    
    
    
    
class DrugGCN(nn.Module):
    def __init__(self, dim_embedding=128, activation=None):
        super(DrugGCN, self).__init__()
        self.init_transform = nn.Linear(74, dim_embedding, bias=False)
        self.gnns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.reslins = nn.ModuleList()
        for i in range(3):
            self.gnns.append(GCNConv(dim_embedding, dim_embedding))
            self.norms.append(nn.BatchNorm1d(dim_embedding))
            self.reslins.append(nn.Linear(dim_embedding, dim_embedding))
        # self.gnns = GCN(dim_embedding, dim_embedding, num_layers=3,
        #                 act="relu", norm="batchnorm")
    def forward(self, g):
        x = g.x
        x = self.init_transform(x)
        edge_index = g.edge_index
        for i in range(3):
            x_new = self.norms[i](self.gnns[i](x, edge_index)).relu()
            x = x_new + self.reslins[i](x)
        
        # x = self.gnns(x, edge_index)
        dx, _ = to_dense_batch(x, g.batch)
        return dx


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, q_mask=None, v_mask=None, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            # Add numerical stability
            if torch.isnan(v_).any():
                v_ = torch.nan_to_num(v_)
            if torch.isnan(q_).any():
                q_ = torch.nan_to_num(q_)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            # Add numerical stability
            if torch.isnan(v_).any():
                v_ = torch.nan_to_num(v_)
            if torch.isnan(q_).any():
                q_ = torch.nan_to_num(q_)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        
        # ===== 🔍 用 v_mask 屏蔽分子图 pad 位置 =====
        if v_mask is not None and softmax:
            # v_mask: [B, v_num], 1=有效，0=pad
            mask = (v_mask == 0).unsqueeze(1).unsqueeze(3)  # [B, 1, v_num, 1]
            neg = -1e4 if att_maps.dtype in (torch.float16, torch.bfloat16) else -1e9
            att_maps = att_maps.masked_fill(mask, neg)
        
        # ===== 🔍 用 q_mask 屏蔽蛋白 pad 位置 =====
        if q_mask is not None and softmax:
            # q_mask: [B, q_num], 1=有效，0=pad
            mask = (q_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, q_num]
            neg = -1e4 if att_maps.dtype in (torch.float16, torch.bfloat16) else -1e9
            att_maps = att_maps.masked_fill(mask, neg)
        
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2) + 1e-10
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits
