#!/bin/bash

# ============================================================================
# DiffB110 Environment Setup Script - Create dif2 Environment
# ============================================================================
# This script creates a clean, optimized environment for DiffB110 project
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     DiffB110 Environment Setup - Creating dif2 Environment     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Step 1: Check if mamba is available
# ============================================================================
echo -e "${YELLOW}[1/6] Checking mamba installation...${NC}"

if ! command -v mamba &> /dev/null; then
    echo -e "${RED}✗ mamba not found. Installing mamba...${NC}"
    conda install -c conda-forge mamba -y
fi

MAMBA_VERSION=$(mamba --version)
echo -e "${GREEN}✓ mamba available: $MAMBA_VERSION${NC}"

# ============================================================================
# Step 2: Remove old dif2 environment if it exists
# ============================================================================
echo ""
echo -e "${YELLOW}[2/6] Checking for existing dif2 environment...${NC}"

if mamba env list | grep -q "^dif2 "; then
    echo -e "${YELLOW}⚠ dif2 environment already exists. Removing...${NC}"
    mamba env remove -n dif2 -y
    echo -e "${GREEN}✓ Old dif2 environment removed${NC}"
else
    echo -e "${GREEN}✓ No existing dif2 environment${NC}"
fi

# ============================================================================
# Step 3: Create base environment with Python 3.10
# ============================================================================
echo ""
echo -e "${YELLOW}[3/6] Creating base dif2 environment with Python 3.10...${NC}"

mamba create -n dif2 python=3.10 -c conda-forge -y

echo -e "${GREEN}✓ Base environment created${NC}"

# ============================================================================
# Step 4: Install core dependencies via mamba (faster than pip)
# ============================================================================
echo ""
echo -e "${YELLOW}[4/6] Installing core dependencies via mamba...${NC}"

# Core scientific packages
mamba install -n dif2 \
    numpy scipy pandas scikit-learn \
    matplotlib seaborn plotly \
    -c conda-forge -y

echo -e "${GREEN}✓ Scientific packages installed${NC}"

# PyTorch with CUDA support
echo -e "${YELLOW}   Installing PyTorch with CUDA...${NC}"
mamba install -n dif2 \
    pytorch::pytorch pytorch::pytorch-cuda=12.1 pytorch::torchvision pytorch::torchaudio \
    -c pytorch -c conda-forge -y

echo -e "${GREEN}✓ PyTorch installed${NC}"

# Chemistry packages
echo -e "${YELLOW}   Installing chemistry packages...${NC}"
mamba install -n dif2 \
    rdkit biopython openbabel \
    -c conda-forge -y

echo -e "${GREEN}✓ Chemistry packages installed${NC}"

# Molecular dynamics and quantum chemistry
echo -e "${YELLOW}   Installing molecular dynamics packages...${NC}"
mamba install -n dif2 \
    ase \
    -c conda-forge -y

echo -e "${GREEN}✓ Molecular dynamics packages installed${NC}"

# Development and utilities
echo -e "${YELLOW}   Installing development tools...${NC}"
mamba install -n dif2 \
    git pip setuptools wheel \
    tqdm pyyaml requests \
    -c conda-forge -y

echo -e "${GREEN}✓ Development tools installed${NC}"

# ============================================================================
# Step 5: Install pip packages (those not available in conda)
# ============================================================================
echo ""
echo -e "${YELLOW}[5/6] Installing pip packages...${NC}"

# Activate environment for pip install
source activate dif2

# Core ML packages
pip install --upgrade \
    pytorch-lightning>=2.0.0 \
    ray[tune]==2.41.0 \
    wandb>=0.13.1 \
    tensordict>=0.2.0

echo -e "${GREEN}✓ PyTorch Lightning, Ray, and wandb installed${NC}"

# Reinforcement learning and training
pip install --upgrade \
    trl==0.12.1 \
    accelerate>=0.34.0

echo -e "${GREEN}✓ RL and training packages installed${NC}"

# Additional utilities
pip install --upgrade \
    easydict \
    loguru \
    pyscf \
    cffi \
    qcelemental

echo -e "${GREEN}✓ Additional utilities installed${NC}"

# ============================================================================
# Step 6: Verify installation
# ============================================================================
echo ""
echo -e "${YELLOW}[6/6] Verifying installation...${NC}"

python3 << 'VERIFY_EOF'
import sys
print("\n✓ Python version:", sys.version.split()[0])

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import pytorch_lightning as pl
    print(f"✓ PyTorch Lightning {pl.__version__}")
except ImportError as e:
    print(f"✗ PyTorch Lightning: {e}")

try:
    import ray
    print(f"✓ Ray {ray.__version__}")
except ImportError as e:
    print(f"✗ Ray: {e}")

try:
    import rdkit
    print(f"✓ RDKit {rdkit.__version__}")
except ImportError as e:
    print(f"✗ RDKit: {e}")

try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import scipy
    print(f"✓ SciPy {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy: {e}")

try:
    import ase
    print(f"✓ ASE {ase.__version__}")
except ImportError as e:
    print(f"✗ ASE: {e}")

try:
    import wandb
    print(f"✓ Weights & Biases {wandb.__version__}")
except ImportError as e:
    print(f"✗ Weights & Biases: {e}")

print("\n✓ All core packages verified!")
VERIFY_EOF

echo ""
echo -e "${GREEN}✓ Installation verification complete${NC}"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Setup Complete!                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Environment: dif2${NC}"
echo -e "${GREEN}Python: 3.10${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Activate environment:"
echo "     conda activate dif2"
echo ""
echo "  2. Navigate to project:"
echo "     cd /home/md666/data/disk6/DiffB110"
echo ""
echo "  3. Install project in development mode:"
echo "     pip install -e ."
echo ""
echo "  4. Run training with DDP:"
echo "     ./ddpo/configs/run_training_ddp.sh"
echo ""
echo -e "${YELLOW}To compare with old environment:${NC}"
echo "  conda activate dif   # Old environment"
echo "  conda activate dif2  # New environment"
echo ""
