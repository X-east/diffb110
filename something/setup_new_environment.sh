#!/bin/bash

# ============================================================================
# DiffB110 环境自动配置脚本
# ============================================================================
# 自动创建新的本地环境，避免权限问题
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="/home/md666/data/disk6/DiffB110"
ENVS_DIR="$PROJECT_DIR/.envs"
ENV_NAME="dif_new"
ENV_PATH="$ENVS_DIR/$ENV_NAME"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        DiffB110 环境自动配置脚本                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# 步骤 1: 创建环境目录
# ============================================================================
echo -e "${YELLOW}[1/8] 创建环境目录...${NC}"
mkdir -p "$ENVS_DIR"
echo -e "${GREEN}✓ 环境目录: $ENVS_DIR${NC}"

# ============================================================================
# 步骤 2: 配置 conda
# ============================================================================
echo ""
echo -e "${YELLOW}[2/8] 配置 conda...${NC}"

# 检查是否已配置
if grep -q "DiffB110 项目本地环境" ~/.condarc 2>/dev/null; then
    echo -e "${GREEN}✓ conda 已配置${NC}"
else
    cat >> ~/.condarc << 'CONDARC'

# DiffB110 项目本地环境
envs_dirs:
  - /home/md666/data/disk6/DiffB110/.envs
CONDARC
    echo -e "${GREEN}✓ conda 配置已添加${NC}"
fi

# ============================================================================
# 步骤 3: 创建环境
# ============================================================================
echo ""
echo -e "${YELLOW}[3/8] 创建 Python 3.10 环境...${NC}"

if [ -d "$ENV_PATH" ]; then
    echo -e "${YELLOW}⚠ 环境已存在，跳过创建${NC}"
else
    CONDA_NO_PLUGINS=true conda create \
        -p "$ENV_PATH" \
        python=3.10 -y > /dev/null 2>&1
    echo -e "${GREEN}✓ 环境已创建: $ENV_PATH${NC}"
fi

# ============================================================================
# 步骤 4: 激活环境并安装 PyTorch
# ============================================================================
echo ""
echo -e "${YELLOW}[4/8] 安装 PyTorch for CUDA 12.1...${NC}"

source activate "$ENV_PATH"

conda install pytorch::pytorch pytorch::pytorch-cuda=12.1 \
    -c pytorch -y > /dev/null 2>&1

echo -e "${GREEN}✓ PyTorch 已安装${NC}"

# ============================================================================
# 步骤 5: 安装科学计算包
# ============================================================================
echo ""
echo -e "${YELLOW}[5/8] 安装科学计算包...${NC}"

conda install numpy scipy pandas scikit-learn \
    matplotlib seaborn rdkit biopython openbabel ase \
    -c conda-forge -y > /dev/null 2>&1

echo -e "${GREEN}✓ 科学计算包已安装${NC}"

# ============================================================================
# 步骤 6: 安装 pip 包
# ============================================================================
echo ""
echo -e "${YELLOW}[6/8] 安装 pip 包...${NC}"

pip install --upgrade --quiet \
    pytorch-lightning>=2.0.0 \
    ray[tune]==2.41.0 \
    wandb>=0.13.1 \
    tensordict>=0.2.0 \
    trl==0.12.1 \
    accelerate>=0.34.0 \
    easydict loguru pyscf cffi qcelemental

echo -e "${GREEN}✓ pip 包已安装${NC}"

# ============================================================================
# 步骤 7: 验证安装
# ============================================================================
echo ""
echo -e "${YELLOW}[7/8] 验证安装...${NC}"

python << 'VERIFY_EOF'
import sys
try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    print(f"  ✓ GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)
VERIFY_EOF

# ============================================================================
# 步骤 8: 显示使用说明
# ============================================================================
echo ""
echo -e "${YELLOW}[8/8] 配置完成！${NC}"
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    使用说明                                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}环境信息:${NC}"
echo "  名称: $ENV_NAME"
echo "  路径: $ENV_PATH"
echo ""
echo -e "${GREEN}激活环境:${NC}"
echo "  conda activate $ENV_PATH"
echo ""
echo -e "${GREEN}运行训练:${NC}"
echo "  cd $PROJECT_DIR"
echo "  ./ddpo/configs/run_training_ddp.sh"
echo ""
echo -e "${GREEN}或直接运行:${NC}"
echo "  conda activate $ENV_PATH"
echo "  cd $PROJECT_DIR"
echo "  python ddpo/main_diffsbdd_lightning.py \\"
echo "    --config_path ddpo/configs/ddp_8gpu_production.yaml"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
