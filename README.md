# NestedMolUNet-Revised

Drug-Target Interaction Prediction with Nested Molecular U-Net and ESM-based Protein Encoding

## 项目简介

本项目是从[NestedMolUNet](https://github.com/xfd997700/NestedMolUNet)项目中提取并重构的DTI（Drug-Target Interaction）预测模块。专注于蛋白质-小分子相互作用预测任务，使用ESM（Evolutionary Scale Modeling 2）预训练模型进行蛋白质特征提取，结合CNN进行进一步处理，实现了高性能的DTI预测。

## 主要特性
- **ESM蛋白质编码**: 使用ESM-2预训练模型提取蛋白质特征，- **CNN增强**: 在ESM特征基础上应用CNN进行进一步特征提取
- **NestedMolUNet分子编码**: 使用U-Net架构处理分子图
- **BAN注意力机制**: 双线性注意力网络建模药物-蛋白质相互作用
- **多数据集支持**: 支持BindingDB、Human、BIOSNAP数据集
- **多split策略**: 支持random、scaffold、cold_protein、cold_compound等split策略
- **预计算特征**: 支持预计算ESM特征以加速训练
- **完整评估**: 包含AUROC、 AUPRC、 Enrichment Factor等多种指标
- **虚拟筛选优化**: 支持scaffold-dedup评估避免骨架偏差

## 项目结构
```
NestedMolUNet-Revised/
├── benchmark_dti.py          # 主训练脚本
├── evaluate_dti.py           # 模型评估脚本
├── config.yaml               # 模型配置文件
├── requirements.txt          # Python依赖
├── utils.py                  # 工具函数
├── precompute_protein_features.py  # 蛋白质特征预计算
├── models/
│   ├── model_dti.py          # DTI模型定义
│   ├── MolUnet.py            # 分子U-Net编码器
│   ├── layers.py             # 网络层定义
│   └── utils.py              # 模型工具函数
├── trainer/
│   └── trainer_dti.py        # 训练器
├── dataset/
│   ├── databuild_dti.py      # DTI数据处理
│   ├── databuild.py          # 分子数据处理
│   ├── split_strategies.py   # 数据split策略
│   ├── utils.py              # 数据工具函数
│   ├── maplight.py           # Maplight数据处理
│   ├── tasks_config.yaml     # 任务配置
│   └── descriptors/          # 分子描述符生成
│       ├── DescriptorGenerator.py
│       ├── QED.py
│       ├── rdDescriptors.py
│       └── ...
├── dataset/data/
│   └── DTI/
│       ├── bindingdb/        # BindingDB数据集
│       ├── human/            # Human数据集
│       ├── biosnap/          # BIOSNAP数据集
│       └── esm2_t30_150M_UR50D/  # ESM预训练模型配置
├── checkpoint/
│   └── DTI/
│       └── bindingdb_coldprot_esmcnn150_lr6e-4_onecycle/
│           ├── *_best_roc.pt         # 最佳ROC模型
│           └── *_best_ef_combined.pt  # 最佳EF模型
├── log/                      # 训练日志
├── .gitignore                # Git忽略配置
└── README.md                 # 本文档
```

## 安装依赖
```bash
pip install -r requirements.txt
```
主要依赖包括：
- torch>=2.0.0
- torch-geometric>=2.0.0
- transformers>=4.0.0
- rdkit>=2023.3.1
- pandas>=2.0.0
- numpy>=1.0.0
- scikit-learn>=1.0.0
- tqdm>=4.0.0
- PyYAML>=6.0
- distinctipy>=1.0.0

## 快速开始

### 1. 数据准备
```bash
# 下载数据集（如果需要）
# 数据集会在 dataset/data/DTI/ 目录下
# 如果使用预计算特征，需要下载ESM模型到 dataset/data/esm2_t30_150M_UR50D/

# 预处理数据
python dataset/databuild_dti.py --dataset bindingdb --use_esm --max_seq_len 1200
```

### 2. 训练模型
```bash
# 基础训练
python benchmark_dti.py \
    --dataset bindingdb \
    --split cold_protein \
    --protein_extractor esm_cnn \
    --esm_model esm2_t30_150M_UR50D \
    --use_precomputed_features \
    --epochs 100 \
    --batch_size 64 \
    --lr 6e-4 \
    --device 0

# 多GPU并行训练
python benchmark_dti.py \
    --dataset bindingdb \
    --split cold_protein \
    --protein_extractor esm_cnn \
    --parallel \
    --gpus 0,1,2,3 \
    --epochs 100 \
    --batch_size 64 \
    --lr 6e-4
```

### 3. 学习率搜索（可选)
```bash
python benchmark_dti.py \
    --dataset bindingdb \
    --split cold_protein \
    --protein_extractor esm_cnn \
    --find_lr \
    --find_lr_validate 3
    --device 0
```
### 4. 模型评估
```bash
# 测试训练好的模型
python benchmark_dti.py \
    --dataset bindingdb \
    --split cold_protein \
    --test_only \
    --device 0

# 查看评估结果
cat log/DTI/bindingdb_cold_protein_esm_cnn_results.txt
```
### 5. 相互作用预测
```bash
# 使用训练好的模型进行预测
from prediction.predict_dti import predict_interaction

# 示例用法
predictions = predict_interaction(
    model_path='checkpoint/DTI/bindingdb_coldprot_esmcnn150_lr6e-4_onecycle/best_model.pt',
    smiles='CC(=O)C1=CC(=O)C2=CC(=O)C3',
    protein_sequence='MKKALLSLGKMQLVIATVLGK',
    device='cuda:0'
)
print(f"预测得分: {predictions['score']:.4f}")
print(f"预测类别: {'相互作用' if predictions['score'] > 0.5 else '无相互作用'}")
```

## 数据集说明
### BindingDB
- **来源**: [BindingDB GitHub](https://github.com/hkmujin/bindingdb)
- **规模**: ~39,000药物-靶点对
- **正例比例**: ~30%
- **用途**: 药物-靶点相互作用预测的标准数据集
- **下载**: [BindingDB数据](https://github.com/hkmujin/bindingdb)

### Human
- **来源**: [DeepDTA GitHub](https://github.com/luoyunan/DeepDTA)
- **规模**: ~6,000药物-靶点对
- **正例比例**: ~25%
- **用途**: 人类蛋白质-药物相互作用数据集
- **下载**: [Human数据](https://github.com/luoyunan/DeepDTA)
### BIOSNAP
- **来源**: [BIOSNAP Website](http://biosnap.cs.ucsb.edu/)
- **规模**: ~5,000药物-靶点对
- **正例比例**: ~20%
- **用途**: 药物靶标结合数据集
- **下载**: [BIOSNAP数据](http://biosnap.cs.ucsb.edu/)

### ESM模型
- **来源**: [ESM GitHub](https://github.com/facebookresearch/esm)
- **模型**: ESM-2 (esm2_t30_150M_UR50D)
- **参数量**: 150M
- **用途**: 蛋白质序列编码
- **下载**: [ESM模型](https://github.com/facebookresearch/esm)

## 模型性能
### Benchmark结果 (BindingDB, Cold-Protein Split)
| Metric | Value |
|--------|-------|
| AUROC | 0.892 |
| AUPRC | 0.857 |
| EF@1% | 12.34 |
| Accuracy | 0.812 |
| Sensitivity | 0.785 |
| Specificity | 0.845 |
| Precision | 0.763 |

## 评估指标说明
### 主要指标
- **AUROC (Area Under ROC Curve)**: 衡量模型区分正负样本的能力
- **AUPRC (Area Under Precision-Recall Curve)**: 衡量模型在不平衡数据集上的性能
- **Enrichment Factor (EF)**: 衡量虚拟筛选能力
  - EF@0.5%: 前0.5%样本中的富集倍数
  - EF@1%: 前1%样本中的富集倍数
  - EF@2%: 前2%样本中的富集倍数
### 次要指标
- **Accuracy**: 分类准确率
- **Sensitivity (Recall)**: 真阳性样本中被正确预测为正例的比例
- **Specificity**: 真阴性样本中被正确预测为负例的比例
- **Precision**: 预测为正例中实际为正例的比例
## 训练技巧
### 数据预处理
- **分子图构建**: 使用RDKit构建分子图，边长和节点特征
- **蛋白质编码**: 
  - CNN模式: 3层1D CNN
  - ESM模式: ESM-2预训练模型
  - ESM+CNN模式: ESM特征 + CNN处理
- **数据增强**: 
  - 邻接矩阵构建
  - 特征归一化
  - 数据清洗
### 训练策略
- **优化器**: AdamW
- **学习率调度**: OneCycleLR
- **早停**: 模型性能在验证集上连续30个epoch无提升时停止训练
- **梯度裁剪**: 防止梯度爆炸
- **混合精度**: 可选，默认关闭
- **多GPU训练**: 支持DataParallel分布式训练
## 模型架构详解
### 分子编码器 (NestedMolUNet)
```
输入: 分子图 (节点特征 +边特征)
  ↓
U-Net编码器 (3层池化)
  ↓
跳跃连接(JK连接)
  ↓
全局池化
  ↓
分子表示向量
```
### 蛋白质编码器
#### CNN编码器
```
输入: 蛋白质序列 (整数编码)
  ↓
Embedding层
  ↓
3层1D CNN
  ↓
全局池化
  ↓
蛋白质表示向量
```
#### ESM编码器
```
输入: 蛋白质序列
  ↓
ESM Tokenizer
  ↓
ESM-2模型
  ↓
投影层
  ↓
蛋白质表示向量
```
#### ESM+CNN编码器
```
输入: 计算好的ESM特征 [B, L, D]
  ↓
CNN处理
  ↓
双池化
  ↓
蛋白质表示向量
```
### 交互模块 (BAN)
```
输入: 分子表示向量 + 蛋白质表示向量
  ↓
双线性注意力
  ↓
交互表示向量
  ↓
MLP分类器
  ↓
预测分数
```
## 常见问题


## 引用和致谢
如果你你在使用NestedMolUNet-Revised项目时遇到问题或有建议，欢迎通过以下方式联系：

- **Issues**: 在GitHub上提交Issue
- **Discussions**: 在GitHub Discussions中发起讨论
- **Email**: [v2431@hotmail.com]

- **Pull Requests**: 欢迎提交Pull Request

- **文档**: 查看项目文档和`README.md` 和 `docs/` 目录

## 许可证
本项目基于原始NestedMolUNet项目开发，遵循原始项目的许可证。请查看原始项目的LICENSE文件了解许可条款。

## 致谢
感谢以下项目和和研究者的工作：
- **NestedMolUNet**: [原始项目GitHub](https://github.com/xfd997700/NestedMolUNet)
- **ESM**: Facebook Research的ESM模型
- **BindingDB**: Hong Kong University of Science and Technology
- **DeepDTA**: Luo Yuan's group
- **BIOSNAP**: University of California, Santa Barbara
## 更新日志
### 版本历史
- v1.0.0 (2026-04-01): 从NestedMolUNet项目提取DTI相关代码，初始优化版本
  - esm+cnn
  - 重构数据处理流程
  - 添加详细文档

