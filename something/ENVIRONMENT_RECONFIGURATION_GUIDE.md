#!/bin/bash

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║              DiffB110 环境重新配置 - 详细方案                              ║
╚════════════════════════════════════════════════════════════════════════════╝

你的系统有以下问题需要解决：
  1. PyTorch CUDA 版本不匹配 (11.8 vs 12.0)
  2. 系统权限配置损坏 (sudo 无法使用)
  3. Conda 多进程权限问题

本文档提供 3 个完整的环境配置方案。

════════════════════════════════════════════════════════════════════════════

方案 1: 本地项目环境（推荐 - 无需 sudo）
════════════════════════════════════════════════════════════════════════════

优点:
  ✓ 无需 sudo 权限
  ✓ 项目级隔离
  ✓ 不影响全局环境
  ✓ 易于管理

缺点:
  ✗ 环境文件较大 (~5GB)
  ✗ 每个项目需要单独配置

步骤:

1. 创建本地环境目录
   ─────────────────
   mkdir -p /home/md666/data/disk6/DiffB110/.envs

2. 配置 conda 使用本地目录
   ──────────────────────────
   cat >> ~/.condarc << 'CONDARC'

# DiffB110 项目本地环境
envs_dirs:
  - /home/md666/data/disk6/DiffB110/.envs
  - /home/md666/miniforge3/envs
CONDARC

3. 创建新环境（使用 --no-plugins 避免权限问题）
   ──────────────────────────────────────────────
   CONDA_NO_PLUGINS=true conda create \
     -p /home/md666/data/disk6/DiffB110/.envs/dif_new \
     python=3.10 -y

4. 激活环境
   ────────
   conda activate /home/md666/data/disk6/DiffB110/.envs/dif_new

5. 安装 PyTorch for CUDA 12.1
   ──────────────────────────
   conda install pytorch::pytorch pytorch::pytorch-cuda=12.1 \
     -c pytorch -y

6. 安装其他依赖
   ────────────
   conda install numpy scipy pandas scikit-learn \
     matplotlib seaborn rdkit biopython openbabel ase \
     -c conda-forge -y

7. 安装 pip 包
   ──────────
   pip install --upgrade \
     pytorch-lightning>=2.0.0 \
     ray[tune]==2.41.0 \
     wandb>=0.13.1 \
     tensordict>=0.2.0 \
     trl==0.12.1 \
     accelerate>=0.34.0 \
     easydict loguru pyscf cffi qcelemental

8. 验证 GPU
   ────────
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

9. 运行训练
   ────────
   cd /home/md666/data/disk6/DiffB110
   ./ddpo/configs/run_training_ddp.sh

════════════════════════════════════════════════════════════════════════════

方案 2: 使用 Mamba 创建环境（快速 - 无需 sudo）
════════════════════════════════════════════════════════════════════════════

优点:
  ✓ 比 conda 快 10 倍
  ✓ 无需 sudo
  ✓ 自动解决依赖冲突

缺点:
  ✗ 需要安装 mamba

步骤:

1. 安装 mamba（如果未安装）
   ────────────────────────
   conda install -c conda-forge mamba -y

2. 创建环境
   ────────
   mamba create -n dif_mamba python=3.10 \
     pytorch::pytorch pytorch::pytorch-cuda=12.1 \
     numpy scipy pandas scikit-learn \
     matplotlib seaborn rdkit biopython openbabel ase \
     -c pytorch -c conda-forge -y

3. 激活环境
   ────────
   conda activate dif_mamba

4. 安装 pip 包
   ──────────
   pip install --upgrade \
     pytorch-lightning ray[tune] wandb tensordict \
     trl accelerate easydict loguru pyscf cffi qcelemental

5. 验证
   ────
   python -c "import torch; print(torch.cuda.is_available())"

6. 运行训练
   ────────
   cd /home/md666/data/disk6/DiffB110
   ./ddpo/configs/run_training_ddp.sh

════════════════════════════════════════════════════════════════════════════

方案 3: Docker 容器（最可靠 - 完全隔离）
════════════════════════════════════════════════════════════════════════════

优点:
  ✓ 完全隔离环境
  ✓ 避免所有权限问题
  ✓ 自动处理 CUDA 版本
  ✓ 易于复现

缺点:
  ✗ 需要安装 Docker
  ✗ 镜像文件较大 (~10GB)

步骤:

1. 创建 Dockerfile
   ───────────────
   cat > /home/md666/data/disk6/DiffB110/Dockerfile << 'DOCKERFILE'
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# 安装 Python 和基础工具
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    build-essential libopenbabel-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 复制项目
COPY . /workspace

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install \
    pytorch-lightning>=2.0.0 \
    ray[tune]==2.41.0 \
    wandb>=0.13.1 \
    tensordict>=0.2.0 \
    trl==0.12.1 \
    accelerate>=0.34.0 \
    numpy scipy pandas scikit-learn \
    matplotlib seaborn rdkit biopython openbabel ase \
    easydict loguru pyscf cffi qcelemental

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_LAUNCH_BLOCKING=0

# 默认命令
CMD ["python", "ddpo/main_diffsbdd_lightning.py", \
     "--config_path", "ddpo/configs/ddp_8gpu_production.yaml"]
DOCKERFILE

2. 构建镜像
   ────────
   cd /home/md666/data/disk6/DiffB110
   docker build -t diffb110:latest .

3. 运行容器（使用 GPU）
   ──────────────────
   docker run --gpus all \
     -v $(pwd):/workspace \
     -it diffb110:latest

4. 或运行特定命令
   ──────────────
   docker run --gpus all \
     -v $(pwd):/workspace \
     diffb110:latest \
     python ddpo/main_diffsbdd_lightning.py \
     --config_path ddpo/configs/ddp_8gpu_production.yaml

════════════════════════════════════════════════════════════════════════════

快速对比表
════════════════════════════════════════════════════════════════════════════

特性              | 方案1(本地)  | 方案2(Mamba) | 方案3(Docker)
─────────────────────────────────────────────────────────────────────────
需要 sudo         | ✗           | ✗           | ✗
设置时间          | 10分钟      | 5分钟       | 15分钟
环境大小          | ~5GB        | ~5GB        | ~10GB
隔离性            | 中          | 中          | 极高
易用性            | 高          | 高          | 中
推荐度            | ⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐

════════════════════════════════════════════════════════════════════════════

完整配置脚本（方案 1 - 本地环境）
════════════════════════════════════════════════════════════════════════════

#!/bin/bash
set -e

PROJECT_DIR="/home/md666/data/disk6/DiffB110"
ENVS_DIR="$PROJECT_DIR/.envs"
ENV_NAME="dif_new"

echo "=== DiffB110 环境配置 ==="
echo ""

# 1. 创建目录
echo "1. 创建环境目录..."
mkdir -p "$ENVS_DIR"

# 2. 配置 conda
echo "2. 配置 conda..."
cat >> ~/.condarc << 'CONDARC'

# DiffB110 项目本地环境
envs_dirs:
  - /home/md666/data/disk6/DiffB110/.envs
CONDARC

# 3. 创建环境
echo "3. 创建 Python 3.10 环境..."
CONDA_NO_PLUGINS=true conda create \
  -p "$ENVS_DIR/$ENV_NAME" \
  python=3.10 -y

# 4. 激活环境
echo "4. 激活环境..."
source activate "$ENVS_DIR/$ENV_NAME"

# 5. 安装 PyTorch
echo "5. 安装 PyTorch for CUDA 12.1..."
conda install pytorch::pytorch pytorch::pytorch-cuda=12.1 \
  -c pytorch -y

# 6. 安装科学包
echo "6. 安装科学计算包..."
conda install numpy scipy pandas scikit-learn \
  matplotlib seaborn rdkit biopython openbabel ase \
  -c conda-forge -y

# 7. 安装 pip 包
echo "7. 安装 pip 包..."
pip install --upgrade \
  pytorch-lightning>=2.0.0 \
  ray[tune]==2.41.0 \
  wandb>=0.13.1 \
  tensordict>=0.2.0 \
  trl==0.12.1 \
  accelerate>=0.34.0 \
  easydict loguru pyscf cffi qcelemental

# 8. 验证
echo "8. 验证安装..."
python -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()}')
print(f'✓ GPUs: {torch.cuda.device_count()}')
"

echo ""
echo "=== 配置完成！==="
echo ""
echo "激活环境:"
echo "  conda activate $ENVS_DIR/$ENV_NAME"
echo ""
echo "运行训练:"
echo "  cd $PROJECT_DIR"
echo "  ./ddpo/configs/run_training_ddp.sh"

════════════════════════════════════════════════════════════════════════════

推荐步骤
════════════════════════════════════════════════════════════════════════════

1. 选择方案
   ├─ 如果想快速: 选择方案 2 (Mamba)
   ├─ 如果想简单: 选择方案 1 (本地环境)
   └─ 如果想可靠: 选择方案 3 (Docker)

2. 按照步骤执行

3. 验证 GPU
   python -c "import torch; print(torch.cuda.is_available())"

4. 运行训练
   ./ddpo/configs/run_training_ddp.sh

════════════════════════════════════════════════════════════════════════════

故障排除
════════════════════════════════════════════════════════════════════════════

问题: "Permission denied" 错误
解决: 使用 CONDA_NO_PLUGINS=true 前缀

问题: "CUDA not available"
解决: 检查 PyTorch CUDA 版本是否与系统匹配

问题: "conda: command not found"
解决: 运行 source ~/.bashrc 重新加载环境

问题: Docker 无法访问 GPU
解决: 确保安装了 nvidia-docker

════════════════════════════════════════════════════════════════════════════

EOF
