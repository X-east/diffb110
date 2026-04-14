#!/bin/bash
# 2 GPU Test Script with PyTorch Lightning
# Usage: bash scripts/test_2gpu_lightning.sh

set -e  # Exit on error

echo "=========================================="
echo "Starting 2 GPU Lightning DDP Test"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set OpenBabel plugin directory
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Set CUDA devices (only 2 GPUs for testing)
export CUDA_VISIBLE_DEVICES=0,1

# Create test config (copy from 8gpu and modify)
cat > ddpo/configs/ddp_2gpu_test.yaml << 'EOF'
# 2 GPU DDP Test Configuration
name: diffsbdd-ddpo-2gpu-test

model:
  checkpoint: "./DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
  time_step: 500
  device: "cuda"

pocket:
  pdb_path: "./data/8GJJ-C-Cip.pdb"
  ref_ligand: "C:1101"

reward:
  weights:
    docking: 0.7
    druglikeness: 0.3
  docking:
    normalization_factor: -10.0
    center_x: 161.45
    center_y: 169.80
    center_z: 174.23
    size_x: 20.0
    size_y: 20.0
    size_z: 20.0
    exhaustiveness: 8

dataloader:
  sample_group_size: 2
  each_prompt_sample: 4
  epochs: 3  # Only 3 epochs for testing
  epoches: 3
  min_nodes: 10
  max_nodes: 30

train:
  batch_size: 8  # Total batch size (2 GPUs × 4 per GPU)
  per_device_batch_size: 4
  learning_rate: 3.0e-4
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_weight_decay: 1.0e-4
  adam_epsilon: 1.0e-8
  clip_range: 0.0001
  adv_clip_max: 10.0
  max_grad_norm: 1.0
  train_micro_batch_size: 4
  checkpoint_interval: 1
  log_interval: 1

ddp:
  enabled: true
  num_gpus: 2
  strategy: "ddp"
  find_unused_parameters: false
  sync_batchnorm: true

ray:
  num_cpus: 8
  num_gpus: 0

wandb:
  enabled: false
  mode: "disabled"

output:
  dir: "./log/ddp-2gpu-test"
  save_frequency: 1
  metrics_file: "metrics.csv"
  molecules_dir: "molecules"
  verbose: true
EOF

echo "Test config created: ddpo/configs/ddp_2gpu_test.yaml"

# Run training
python ddpo/main_diffsbdd_lightning.py \
    --config_path ddpo/configs/ddp_2gpu_test.yaml \
    --seed 42 \
    --log_level INFO

echo "=========================================="
echo "2 GPU test completed!"
echo "=========================================="
