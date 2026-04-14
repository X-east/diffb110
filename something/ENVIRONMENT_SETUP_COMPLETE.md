# DiffB110 环境重新配置 - 完整指南

## 📋 快速开始

### 最简单的方法（推荐）

运行自动配置脚本：

```bash
bash /home/md666/data/disk6/DiffB110/setup_new_environment.sh
```

这个脚本会：
1. ✅ 创建本地环境目录
2. ✅ 配置 conda
3. ✅ 创建 Python 3.10 环境
4. ✅ 安装 PyTorch for CUDA 12.1
5. ✅ 安装所有依赖
6. ✅ 验证 GPU 检测
7. ✅ 显示使用说明

**预计时间**: 15-20 分钟

---

## 🎯 三个完整方案

### 方案 1: 本地项目环境（推荐）

**优点**:
- ✅ 无需 sudo
- ✅ 项目级隔离
- ✅ 易于管理

**步骤**:
```bash
# 1. 运行自动脚本
bash /home/md666/data/disk6/DiffB110/setup_new_environment.sh

# 2. 激活环境
conda activate /home/md666/data/disk6/DiffB110/.envs/dif_new

# 3. 运行训练
cd /home/md666/data/disk6/DiffB110
./ddpo/configs/run_training_ddp.sh
```

---

### 方案 2: 使用 Mamba（最快）

**优点**:
- ✅ 比 conda 快 10 倍
- ✅ 自动解决依赖冲突

**步骤**:
```bash
# 1. 安装 mamba
conda install -c conda-forge mamba -y

# 2. 创建环境
mamba create -n dif_mamba python=3.10 \
  pytorch::pytorch pytorch::pytorch-cuda=12.1 \
  numpy scipy pandas scikit-learn \
  matplotlib seaborn rdkit biopython openbabel ase \
  -c pytorch -c conda-forge -y

# 3. 激活环境
conda activate dif_mamba

# 4. 安装 pip 包
pip install --upgrade \
  pytorch-lightning ray[tune] wandb tensordict \
  trl accelerate easydict loguru pyscf cffi qcelemental

# 5. 运行训练
cd /home/md666/data/disk6/DiffB110
./ddpo/configs/run_training_ddp.sh
```

---

### 方案 3: Docker 容器（最可靠）

**优点**:
- ✅ 完全隔离
- ✅ 避免所有权限问题
- ✅ 自动处理 CUDA 版本

**步骤**:
```bash
# 1. 创建 Dockerfile（已在项目中）
cd /home/md666/data/disk6/DiffB110

# 2. 构建镜像
docker build -t diffb110:latest .

# 3. 运行容器
docker run --gpus all \
  -v $(pwd):/workspace \
  -it diffb110:latest

# 4. 在容器内运行训练
python ddpo/main_diffsbdd_lightning.py \
  --config_path ddpo/configs/ddp_8gpu_production.yaml
```

---

## 📊 方案对比

| 特性 | 方案1(本地) | 方案2(Mamba) | 方案3(Docker) |
|------|-----------|-----------|-------------|
| 需要 sudo | ✗ | ✗ | ✗ |
| 设置时间 | 15分钟 | 5分钟 | 15分钟 |
| 环境大小 | ~5GB | ~5GB | ~10GB |
| 隔离性 | 中 | 中 | 极高 |
| 易用性 | 高 | 高 | 中 |
| 推荐度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🔍 系统信息

### 硬件
```
✓ 8x NVIDIA RTX 3080 GPU
✓ NVIDIA 驱动 590.48.01 已加载
✓ CUDA 工具链 12.0 已安装
```

### 当前问题
```
✗ PyTorch CUDA 版本不匹配 (11.8 vs 12.0)
✗ 系统权限配置损坏 (sudo 无法使用)
✗ Conda 多进程权限问题
```

### 解决方案
```
✓ 创建本地环境避免权限问题
✓ 安装正确的 PyTorch CUDA 12.1 版本
✓ 使用 CONDA_NO_PLUGINS=true 避免多进程问题
```

---

## 🚀 立即行动

### 推荐（最简单）
```bash
bash /home/md666/data/disk6/DiffB110/setup_new_environment.sh
```

### 或手动步骤
```bash
# 1. 创建环境目录
mkdir -p /home/md666/data/disk6/DiffB110/.envs

# 2. 配置 conda
cat >> ~/.condarc << 'EOF'

envs_dirs:
  - /home/md666/data/disk6/DiffB110/.envs
EOF

# 3. 创建环境
CONDA_NO_PLUGINS=true conda create \
  -p /home/md666/data/disk6/DiffB110/.envs/dif_new \
  python=3.10 -y

# 4. 激活环境
conda activate /home/md666/data/disk6/DiffB110/.envs/dif_new

# 5. 安装 PyTorch
conda install pytorch::pytorch pytorch::pytorch-cuda=12.1 \
  -c pytorch -y

# 6. 安装其他包
conda install numpy scipy pandas scikit-learn \
  matplotlib seaborn rdkit biopython openbabel ase \
  -c conda-forge -y

# 7. 安装 pip 包
pip install --upgrade \
  pytorch-lightning ray[tune] wandb tensordict \
  trl accelerate easydict loguru pyscf cffi qcelemental

# 8. 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 9. 运行训练
cd /home/md666/data/disk6/DiffB110
./ddpo/configs/run_training_ddp.sh
```

---

## 📁 相关文档

| 文档 | 说明 |
|------|------|
| `ENVIRONMENT_RECONFIGURATION_GUIDE.md` | 详细的环境配置指南 |
| `setup_new_environment.sh` | 自动配置脚本 |
| `SYSTEM_ISSUES_SUMMARY.md` | 系统问题诊断 |
| `CUDA_GPU_DIAGNOSTIC.md` | CUDA/GPU 诊断 |
| `CONDA_PERMISSIONS_EXPLAINED.md` | 权限问题分析 |

---

## ✅ 验证清单

配置完成后，检查：

- [ ] 环境已创建: `ls -la /home/md666/data/disk6/DiffB110/.envs/`
- [ ] 环境已激活: `conda info | grep "active environment"`
- [ ] PyTorch 已安装: `python -c "import torch; print(torch.__version__)"`
- [ ] CUDA 可用: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] GPU 检测: `python -c "import torch; print(torch.cuda.device_count())"`
- [ ] 配置文件存在: `ls ddpo/configs/ddp_8gpu_production.yaml`
- [ ] 脚本可执行: `ls -l ddpo/configs/run_training_ddp.sh`

---

## 💡 故障排除

### 问题: "Permission denied" 错误
**解决**: 使用 `CONDA_NO_PLUGINS=true` 前缀

### 问题: "CUDA not available"
**解决**: 检查 PyTorch CUDA 版本是否为 12.1

### 问题: "conda: command not found"
**解决**: 运行 `source ~/.bashrc` 重新加载环境

### 问题: 环境创建失败
**解决**: 尝试方案 2 (Mamba) 或方案 3 (Docker)

---

## 🎯 下一步

1. **选择方案**
   - 推荐: 方案 1 (本地环境) - 运行自动脚本
   - 快速: 方案 2 (Mamba)
   - 可靠: 方案 3 (Docker)

2. **配置环境**
   - 按照选定方案的步骤执行

3. **验证安装**
   - 运行验证命令确保 GPU 可用

4. **运行训练**
   - 执行 `./ddpo/configs/run_training_ddp.sh`

---

**现在你有了完整的环境配置方案，可以立即开始！** 🚀
