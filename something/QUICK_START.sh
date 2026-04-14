#!/bin/bash

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║                     DiffB110 - Quick Reference Card                        ║
╚════════════════════════════════════════════════════════════════════════════╝

🔧 WHAT WAS FIXED
════════════════════════════════════════════════════════════════════════════

✅ GPU Parallelism Bug
   • Problem: Only GPU 0 used, memory overflow
   • Fix: Use main_diffsbdd_lightning.py with DDP
   • Script: ./ddpo/configs/run_training_ddp.sh

✅ Code Bug (UnboundLocalError)
   • File: RLPF-verl_diffusion/verl_diffusion/trainer/ddpo_trainer.py
   • Fix: Initialize metrics = {} at start of fit()

⚠️  Environment Pollution
   • Keep old 'dif' environment for reference
   • Optional: Create clean 'dif2' via: conda create -n dif2 --clone dif -y

════════════════════════════════════════════════════════════════════════════

🚀 RUN TRAINING NOW
════════════════════════════════════════════════════════════════════════════

  conda activate dif
  cd /home/md666/data/disk6/DiffB110
  ./ddpo/configs/run_training_ddp.sh

════════════════════════════════════════════════════════════════════════════

📊 CONFIGURATION
════════════════════════════════════════════════════════════════════════════

  Docking:    UniDock (GPU-accelerated) ✓
  GPUs:       4 (DDP enabled) ✓
  Batch:      256 (32 per GPU) ✓
  Epochs:     200 ✓
  Learning:   3.0e-4 ✓

════════════════════════════════════════════════════════════════════════════

📁 KEY FILES
════════════════════════════════════════════════════════════════════════════

  Training Script:
    ddpo/configs/run_training_ddp.sh (USE THIS)

  Config:
    ddpo/configs/ddp_8gpu_production.yaml

  Fixed Code:
    RLPF-verl_diffusion/verl_diffusion/trainer/ddpo_trainer.py

  Documentation:
    STATUS_REPORT.sh (full report)
    SETUP_DIF2_MANUAL.md (environment setup)

════════════════════════════════════════════════════════════════════════════

💡 MONITOR TRAINING
════════════════════════════════════════════════════════════════════════════

  In another terminal:
    watch -n 1 nvidia-smi

  Check logs:
    tail -f ./exp/diffsbdd-ddpo-production/training_*.log

════════════════════════════════════════════════════════════════════════════

EOF
