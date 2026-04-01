# 项目文件清单（最终版本）

## 核心脚本（根目录）
- `benchmark_dti.py` - 主训练脚本
- `evaluate_dti.py` - 模型评估脚本
- `precompute_protein_features.py` - 蛋白质特征预计算
- `utils.py` - 工具函数

## 配置文件（根目录）
- `config.yaml` - 模型配置
- `requirements.txt` - Python依赖
- `.gitignore` - Git忽略文件

## 文档（根目录）
- `README.md` - 项目说明文档
- `GIT_GUIDE.md` - Git操作指南
- `FILE_LIST.md` - 本文件清单

## 模型文件 (models/)
- `model_dti.py` - DTI模型定义
- `MolUnet.py` - 分子U-Net编码器
- `layers.py` - 网络层定义
- `utils.py` - 模型工具函数
- `__init__.py` - 模块初始化

## 训练器 (trainer/)
- `trainer_dti.py` - DTI训练器（主版本）
- `trainer_dti copy back to beforeOneCycleLR.py` - DTI训练器备份版本（特殊保留）
- `__init__.py` - 模块初始化

## 数据处理 (dataset/)
- `databuild_dti.py` - DTI数据处理
- `databuild.py` - 分子数据处理
- `split_strategies.py` - 数据分割策略
- `utils.py` - 数据工具函数
- `maplight.py` - MapLight相关工具
- `__init__.py` - 模块初始化

## 数据目录 (dataset/data/)
- `DTI/` - DTI数据集（BindingDB, Human, BIOSNAP）
- `esm2_t12_35M_UR50D/` - ESM-2 35M模型
- `esm2_t30_150M_UR50D/` - ESM-2 150M模型
- `esm2_t33_650M_UR50D/` - ESM-2 650M模型

## Checkpoint (checkpoint/)
- `checkpoint/DTI/bindingdb_coldprot_esmcnn150_lr6e-4_onecycle/` - 训练好的模型

## 日志目录 (log/)
- `log/DTI/` - 训练日志存储位置

## 已删除的内容（清理记录）
### 数据目录
- `dataset/data/DDI/` - 药物-药物相互作用数据（与DTI无关）
- `dataset/data/pretrain/` - 预训练数据（与DTI无关）
- `dataset/data/property/` - 分子性质预测数据（与DTI无关）

### 备份文件
- `dataset/databuild_dti copy.py` - 备份文件
- `dataset/databuild_dti copy 2.py` - 备份文件
- `dataset/databuild_pretrain.py` - 预训练相关（与DTI无关）
- `dataset/databuild_property.py` - 性质预测相关（与DTI无关）
- `dataset/databuild_ddi.py` - DDI相关（与DTI无关）
- `dataset/dataset_property.py` - 性质预测相关（与DTI无关）
- `dataset/dataset_pretrain.py` - 预训练相关（与DTI无关）
- `dataset/dataset_admet.py` - ADMET相关（与DTI无关）
- `trainer/trainer_dti copy.py` - 备份文件
- `trainer/trainer_dti copy beforeOneCycleLR.py` - 备份文件
- `trainer/trainer_dti copy260305 beforeEFandTis0.5.py` - 备份文件
- `trainer/trainer_dti  copy afterOneCycleLR.py` - 备份文件
- `trainer/trainer_pretrain.py` - 预训练相关（与DTI无关）
- `trainer/trainer_ddi.py` - DDI相关（与DTI无关）
- `trainer/trainer_property.py` - 性质预测相关（与DTI无关）
- `models/model_dti copy.py` - 备份文件
- `models/model_pretrain.py` - 预训练相关（与DTI无关）
- `models/model_ddi.py` - DDI相关（与DTI无关）
- `models/model_property.py` - 性质预测相关（与DTI无关）

## 项目特点
1. ✅ **精简干净**：只保留DTI相关内容
2. ✅ **无冗余文件**：删除所有备份和非DTI相关文件
3. ✅ **结构清晰**：模块化组织，易于理解和维护
4. ✅ **文档完善**：包含详细的README和操作指南
5. ✅ **可复现性**：包含训练好的checkpoint和完整数据集

## 注意事项
1. 数据集文件较大，建议使用 Git LFS 管理
2. ESM 模型文件需要单独下载或使用Git LFS
3. Checkpoint 文件可以根据需要选择性上传
4. 项目已精简为只包含DTI相关内容，适合协同开发
