#!/bin/bash

# ============================================================================
# DiffB110 Environment Setup - Manual Installation Guide
# ============================================================================
# Due to system permission issues, follow these steps manually
# ============================================================================

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║          DiffB110 Environment Setup - Manual Installation Guide            ║
╚════════════════════════════════════════════════════════════════════════════╝

⚠️  SYSTEM ISSUE DETECTED
════════════════════════════════════════════════════════════════════════════

The system has permission issues preventing automatic environment creation.
Follow these manual steps instead:

════════════════════════════════════════════════════════════════════════════

📋 STEP 1: Create Environment Manually
════════════════════════════════════════════════════════════════════════════

Option A: Using conda (if you have write permissions)
  conda create -n dif2 python=3.10 -y

Option B: Using mamba (faster)
  mamba create -n dif2 python=3.10 -y

Option C: Clone existing environment (fastest)
  conda create -n dif2 --clone dif

════════════════════════════════════════════════════════════════════════════

📋 STEP 2: Activate Environment
════════════════════════════════════════════════════════════════════════════

  conda activate dif2

════════════════════════════════════════════════════════════════════════════

📋 STEP 3: Install Core Dependencies
════════════════════════════════════════════════════════════════════════════

Install PyTorch with CUDA support:
  conda install pytorch::pytorch pytorch::pytorch-cuda=12.1 -c pytorch -y

Install scientific packages:
  conda install numpy scipy pandas scikit-learn matplotlib -y

Install chemistry packages:
  conda install rdkit biopython openbabel -y

Install molecular dynamics:
  conda install ase -y

════════════════════════════════════════════════════════════════════════════

📋 STEP 4: Install pip Packages
════════════════════════════════════════════════════════════════════════════

Core ML packages:
  pip install --upgrade pytorch-lightning>=2.0.0 ray[tune]==2.41.0 wandb tensordict

Reinforcement learning:
  pip install --upgrade trl==0.12.1 accelerate

Utilities:
  pip install --upgrade easydict loguru pyscf cffi qcelemental

════════════════════════════════════════════════════════════════════════════

📋 STEP 5: Verify Installation
════════════════════════════════════════════════════════════════════════════

Run this Python code to verify:

python3 << 'VERIFY'
import torch
import pytorch_lightning as pl
import ray
import rdkit

print(f"✓ PyTorch {torch.__version__}")
print(f"✓ Lightning {pl.__version__}")
print(f"✓ Ray {ray.__version__}")
print(f"✓ RDKit {rdkit.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
print(f"✓ GPUs: {torch.cuda.device_count()}")
VERIFY

════════════════════════════════════════════════════════════════════════════

📋 STEP 6: Install Project
════════════════════════════════════════════════════════════════════════════

Navigate to project and install in development mode:
  cd /home/md666/data/disk6/DiffB110
  pip install -e .

════════════════════════════════════════════════════════════════════════════

🚀 STEP 7: Run Training
════════════════════════════════════════════════════════════════════════════

With the new dif2 environment activated:
  ./ddpo/configs/run_training_ddp.sh

Or manually:
  python ddpo/main_diffsbdd_lightning.py \
    --config_path ddpo/configs/ddp_8gpu_production.yaml

════════════════════════════════════════════════════════════════════════════

📊 ENVIRONMENT COMPARISON
════════════════════════════════════════════════════════════════════════════

Old Environment (dif):
  • May have conflicting packages
  • Used for initial testing
  • Keep for reference

New Environment (dif2):
  • Clean installation
  • Optimized dependencies
  • Recommended for production

════════════════════════════════════════════════════════════════════════════

✅ QUICK REFERENCE - Copy & Paste Commands
════════════════════════════════════════════════════════════════════════════

# Clone from existing environment (fastest)
conda create -n dif2 --clone dif -y

# Activate
conda activate dif2

# Upgrade key packages
pip install --upgrade pytorch-lightning ray[tune] wandb

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Run training
cd /home/md666/data/disk6/DiffB110
./ddpo/configs/run_training_ddp.sh

════════════════════════════════════════════════════════════════════════════

💡 TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════

If you get permission errors:
  • Contact system administrator
  • Or use: conda create -n dif2 --clone dif (clone existing env)

If packages fail to install:
  • Try: pip install --no-cache-dir <package>
  • Or: conda install -c conda-forge <package>

If CUDA is not detected:
  • Check: nvidia-smi
  • Verify: python -c "import torch; print(torch.cuda.is_available())"

════════════════════════════════════════════════════════════════════════════

EOF
