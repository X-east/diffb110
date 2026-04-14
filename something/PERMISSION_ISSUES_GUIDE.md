#!/bin/bash

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║              Conda/Mamba 权限问题 - 根本原因分析与解决方案                  ║
╚════════════════════════════════════════════════════════════════════════════╝

🔍 问题根本原因
════════════════════════════════════════════════════════════════════════════

你遇到的权限问题是由以下原因造成的：

1. Conda/Mamba 目录由 root 拥有
   ├─ /home/md666/miniforge3/envs
   │  └─ 所有者: root (uid 0)
   │  └─ 权限: drwxrwxr-x (775)
   │  └─ 问题: 你的用户 md666 无法写入
   │
   └─ 原因: 可能是系统管理员安装或容器环境问题

2. 系统 sudo 配置损坏
   └─ /etc/sudo.conf 所有者错误
   └─ 导致无法使用 sudo 提升权限

3. 多用户/容器环境
   └─ 可能是 Docker 或多用户系统
   └─ 权限隔离导致冲突

════════════════════════════════════════════════════════════════════════════

✅ 解决方案 1: 修复 Conda 目录权限（推荐）
════════════════════════════════════════════════════════════════════════════

如果你有 sudo 权限，运行：

  sudo chown -R md666:md666 /home/md666/miniforge3
  sudo chown -R md666:md666 /home/md666/.conda
  sudo chown -R md666:md666 /home/md666/.condarc

验证修复：
  ls -ld /home/md666/miniforge3/envs

应该显示:
  drwxrwxr-x md666 md666 ...

════════════════════════════════════════════════════════════════════════════

✅ 解决方案 2: 使用本地环境目录（无需 sudo）
════════════════════════════════════════════════════════════════════════════

在你的项目目录创建本地环境：

  mkdir -p /home/md666/data/disk6/DiffB110/.envs
  
  conda create -p /home/md666/data/disk6/DiffB110/.envs/dif2 \
    python=3.10 -y

激活本地环境：
  conda activate /home/md666/data/disk6/DiffB110/.envs/dif2

或添加到 ~/.condarc：
  envs_dirs:
    - /home/md666/data/disk6/DiffB110/.envs
    - /home/md666/miniforge3/envs

════════════════════════════════════════════════════════════════════════════

✅ 解决方案 3: 重新安装 Miniforge（完全重置）
════════════════════════════════════════════════════════════════════════════

如果权限问题严重，完全重新安装：

  # 1. 备份现有环境（可选）
  conda env export -n dif > /tmp/dif_backup.yml
  
  # 2. 卸载 miniforge
  rm -rf /home/md666/miniforge3
  rm -rf /home/md666/.conda
  rm -rf /home/md666/.condarc
  
  # 3. 重新安装 miniforge
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p /home/md666/miniforge3
  
  # 4. 初始化 shell
  /home/md666/miniforge3/bin/conda init bash
  source ~/.bashrc

════════════════════════════════════════════════════════════════════════════

✅ 解决方案 4: 配置 Conda 使用更好的权限设置
════════════════════════════════════════════════════════════════════════════

编辑或创建 ~/.condarc：

  cat > ~/.condarc << 'CONDARC'
channels:
  - conda-forge
  - defaults

envs_dirs:
  - /home/md666/miniforge3/envs
  - /home/md666/.conda/envs

pkgs_dirs:
  - /home/md666/miniforge3/pkgs
  - /home/md666/.conda/pkgs

auto_activate_base: false
auto_update_conda: false
CONDARC

════════════════════════════════════════════════════════════════════════════

🔧 为 AI 优化的最佳实践
════════════════════════════════════════════════════════════════════════════

为了让 AI 助手更方便地运行命令，建议：

1. 确保用户对 conda 目录有完全权限
   └─ 运行: sudo chown -R $USER:$USER ~/miniforge3

2. 使用本地项目环境
   └─ 在项目目录创建 .envs 文件夹
   └─ 避免全局权限问题

3. 配置 sudoers（如果需要）
   └─ 运行: sudo visudo
   └─ 添加: %sudo ALL=(ALL) NOPASSWD: ALL

4. 使用容器（Docker）
   └─ 完全隔离环境
   └─ 避免权限冲突
   └─ 示例 Dockerfile 见下文

════════════════════════════════════════════════════════════════════════════

📋 快速诊断脚本
════════════════════════════════════════════════════════════════════════════

运行此脚本诊断权限问题：

#!/bin/bash

echo "=== Conda 权限诊断 ==="
echo ""

echo "1. Miniforge 目录所有者:"
ls -ld /home/md666/miniforge3

echo ""
echo "2. Envs 目录权限:"
ls -ld /home/md666/miniforge3/envs

echo ""
echo "3. 当前用户:"
whoami

echo ""
echo "4. 用户组:"
groups

echo ""
echo "5. 写入测试:"
if touch /home/md666/miniforge3/envs/.test 2>/dev/null; then
    rm /home/md666/miniforge3/envs/.test
    echo "✓ 可以写入"
else
    echo "✗ 无法写入 (权限问题)"
fi

echo ""
echo "6. Conda 配置:"
conda config --show envs_dirs

════════════════════════════════════════════════════════════════════════════

🐳 Docker 解决方案（最干净）
════════════════════════════════════════════════════════════════════════════

创建 Dockerfile：

FROM mambaorg/micromamba:latest

# 设置工作目录
WORKDIR /workspace

# 复制项目
COPY . /workspace

# 创建环境
RUN micromamba create -n dif2 python=3.10 -y && \
    micromamba run -n dif2 pip install \
    pytorch-lightning ray[tune] wandb tensordict

# 激活环境
ENV PATH /opt/conda/envs/dif2/bin:$PATH

# 运行训练
CMD ["python", "ddpo/main_diffsbdd_lightning.py", \
     "--config_path", "ddpo/configs/ddp_8gpu_production.yaml"]

构建和运行：
  docker build -t diffb110 .
  docker run --gpus all -v $(pwd):/workspace diffb110

════════════════════════════════════════════════════════════════════════════

📊 权限问题对比
════════════════════════════════════════════════════════════════════════════

方案              | 难度 | 隔离性 | 推荐度 | 说明
─────────────────────────────────────────────────────────────────────────
修复权限          | 低   | 低    | ⭐⭐⭐⭐⭐ | 最快，需要 sudo
本地环境          | 低   | 中    | ⭐⭐⭐⭐  | 无需 sudo，项目级
重新安装          | 中   | 高    | ⭐⭐⭐   | 完全重置
Docker            | 高   | 极高  | ⭐⭐⭐⭐⭐ | 最干净，推荐生产

════════════════════════════════════════════════════════════════════════════

🎯 推荐步骤
════════════════════════════════════════════════════════════════════════════

第1步: 尝试修复权限（如果有 sudo）
  sudo chown -R $USER:$USER ~/miniforge3

第2步: 验证
  conda create -n test_env python=3.10 -y

第3步: 如果成功，删除测试环境
  conda env remove -n test_env -y

第4步: 现在可以正常使用
  conda create -n dif2 --clone dif -y

════════════════════════════════════════════════════════════════════════════

EOF
