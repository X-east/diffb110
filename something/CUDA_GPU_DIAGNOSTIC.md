#!/bin/bash

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║                      CUDA/GPU 问题诊断报告                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

🔴 问题检测
════════════════════════════════════════════════════════════════════════════

1. nvidia-smi 失败
   └─ 错误: "couldn't communicate with the NVIDIA driver"
   └─ 原因: GPU驱动未安装或未运行

2. CUDA 工具链存在
   └─ nvcc 位置: /usr/bin/nvcc
   └─ 但驱动不可用

3. PyTorch 无法检测 GPU
   └─ CUDA available: False
   └─ Device count: 0
   └─ 错误代码: 304 (OS call failed)

════════════════════════════════════════════════════════════════════════════

🔍 可能的原因
════════════════════════════════════════════════════════════════════════════

1. GPU 驱动未安装
   └─ nvidia-driver 包未安装

2. GPU 驱动已安装但未加载
   └─ 需要重启系统
   └─ 或手动加载驱动模块

3. 容器/虚拟环境问题
   └─ Docker 容器未配置 GPU 支持
   └─ 虚拟机未分配 GPU

4. 硬件问题
   └─ GPU 未连接
   └─ GPU 被禁用

════════════════════════════════════════════════════════════════════════════

✅ 解决方案
════════════════════════════════════════════════════════════════════════════

方案 1: 检查 GPU 硬件
─────────────────────
  lspci | grep -i nvidia
  
  如果没有输出，说明系统没有 NVIDIA GPU

方案 2: 安装/更新 GPU 驱动
─────────────────────────
  # Ubuntu/Debian
  sudo apt update
  sudo apt install nvidia-driver-XXX  # 替换 XXX 为版本号
  
  # 重启系统
  sudo reboot
  
  # 验证
  nvidia-smi

方案 3: 在 Docker 中运行（如果使用容器）
──────────────────────────────────────
  docker run --gpus all -it nvidia/cuda:12.1-runtime-ubuntu22.04

方案 4: 在 CPU 上运行（临时解决方案）
──────────────────────────────────────
  # 修改配置文件
  # 在 ddp_8gpu_production.yaml 中改为:
  
  model:
    device: "cpu"  # 改为 cpu
  
  ddp:
    enabled: false  # 禁用 DDP
    num_gpus: 0

方案 5: 检查虚拟机/容器配置
──────────────────────────
  # 如果在虚拟机中
  - 检查 VM 设置是否启用了 GPU 直通
  - 检查 Hyper-V/VMware 配置
  
  # 如果在 Docker 中
  - 使用 nvidia-docker
  - 或添加 --gpus all 参数

════════════════════════════════════════════════════════════════════════════

🔧 快速诊断命令
════════════════════════════════════════════════════════════════════════════

# 1. 检查 GPU 硬件
lspci | grep -i nvidia

# 2. 检查驱动
nvidia-smi

# 3. 检查 CUDA
nvcc --version

# 4. 检查 PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 5. 检查 CUDA 设备
python -c "import torch; print(torch.cuda.device_count())"

════════════════════════════════════════════════════════════════════════════

📋 当前系统状态
════════════════════════════════════════════════════════════════════════════

✓ CUDA 工具链: 已安装 (/usr/bin/nvcc)
✓ PyTorch: 已安装 (2.0.1+cu118)
✗ GPU 驱动: 未检测到
✗ NVIDIA GPU: 未检测到
✗ CUDA 运行时: 不可用

════════════════════════════════════════════════════════════════════════════

🚀 临时解决方案：在 CPU 上运行
════════════════════════════════════════════════════════════════════════════

如果你想立即运行训练（在 CPU 上），修改配置：

1. 编辑配置文件:
   nano /home/md666/data/disk6/DiffB110/ddpo/configs/ddp_8gpu_production.yaml

2. 修改以下部分:
   
   model:
     device: "cpu"  # 改为 cpu
   
   ddp:
     enabled: false  # 禁用 DDP
     num_gpus: 0

3. 运行训练:
   conda activate dif
   cd /home/md666/data/disk6/DiffB110
   python ddpo/main_diffsbdd_lightning.py \
     --config_path ddpo/configs/ddp_8gpu_production.yaml

⚠️  注意: CPU 训练会非常慢（可能需要数小时/数天）

════════════════════════════════════════════════════════════════════════════

💡 建议
════════════════════════════════════════════════════════════════════════════

1. 首先检查硬件
   lspci | grep -i nvidia

2. 如果有 GPU，安装驱动
   sudo apt install nvidia-driver-XXX

3. 如果没有 GPU，考虑：
   ├─ 使用云 GPU (AWS, GCP, Azure)
   ├─ 使用本地 GPU 工作站
   └─ 或在 CPU 上运行（仅用于测试）

════════════════════════════════════════════════════════════════════════════

EOF
