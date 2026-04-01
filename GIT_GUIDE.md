# Git 操作指南指南

## 1. 初始化仓库

如果这是一个新项目，请按以下步骤操作：

### 方法一：使用 GitHub 硏页创建仓库（推荐）

1. 在 GitHub 上创建新仓库 `NestedMolUNet-Revised`
2. 不要初始化 README、 LICENSE 緻加 .gitignore
3. 上传代码:
```bash
git init
git add .
git commit -m "Initial commit: NestedMolUNet DTI project"
git branch -M main
git remote add origin https://github.com/your-username/NestedMolUNet-Revised.git
git push -u origin main
```

### 方法二:从已有项目目录初始化（如果已有 .git 目录）

```bash
cd /mnt/nvme/wlz2025/NestedMolUNet-Revised
git init
git add .
git commit -m "Initial commit: NestedMolUNet DTI project"
git branch -M main
git remote add origin https://github.com/your-username/NestedMolUNet-Revised.git
git push -u origin main
```

## 2. 日常更新流程

当你修改代码后，按以下步骤更新到 GitHub:

```bash
# 查看当前状态
git status

# 添加所有修改的文件
git add .

# 提交修改
git commit -m "描述你的修改"

# 推送到远程仓库
git push origin main
```

## 3. 协同工作流程

### 克隆仓库
其他同学可以使用以下命令克隆仓库：

```bash
git clone https://github.com/your-username/NestedMolUNet-Revised.git
cd NestedMolUNet-Revised
```

### 创建分支进行开发
```bash
# 创建新分支
git checkout -b feature/your-feature-name

# 切换到新分支
git checkout feature/your-feature-name

# 进行开发...
# 提交修改
git add .
git commit -m "Add feature: your feature description"

# 推送到远程
git push origin feature/your-feature-name
```

### 创建 Pull Request
当功能开发完成后，创建 Pull Request

1. 在 GitHub 网页上创建 Pull Request
2. 选择基础分支（通常是 main）
3. 添加描述性标题和4. 提交 Pull Request

5. 等待审核

### 解决冲突
如果多人同时修改同一文件，按以下步骤解决冲突

```bash
# 拉取最新代码
git pull origin main

# 创建新分支解决冲突
git checkout -b fix/conflict

# 解决冲突
git add .
git commit -m"Fix conflict in filename"
git push origin fix/conflict

# 创建 Pull Request
git pull-request -m "Fix conflict in filename"
```

## 4. 巻加协作者
在 GitHub 仓库设置中添加协作者

1. 进入仓库 Settings -> Collaborators
2. 添加协作者的 GitHub 用户名
4. 选择权限（通常选择 Write 权限）

## 5. 其他建议

### 添加 .gitignore
建议添加以下内容到 .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb

# 数据文件（根据实际情况调整）
*.pth
*.npy
*.csv
# 但保留必要的 checkpoint
!checkpoint/DTI/*.pt
!log/DTI/detail/*.log

!checkpoint/DTI/*.png

# 临时文件
*.tmp
*.bak

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### 添加 LICENSE
创建 LICENSE 文件说明项目的开源协议

### 添加 CONTRIBUTING.md
创建 CONTRIBUTING.md 说明如何贡献

### 使用 GitHub Projects 管理大型文件
对于大型数据文件（如 ESM 模型），建议使用 Git LFS

### 定期发布版本
使用 GitHub Releases 发布新版本

## 假如需要帮助
如果在操作过程中遇到任何问题，请随时联系项目维护者。

