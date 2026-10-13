#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IQL_WORKTREE="$PROJECT_ROOT/collected_iql_benchmark_100k_260806/iql_worktree"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/launch/diffsbdd_filtered_bc_rank40_allpreferred_8gpu_20epoch_260822.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/analysis/diffsbdd_filtered_bc_rank40_allpreferred_8gpu_20epoch_260822}"
MASTER_PORT="${MASTER_PORT:-29628}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$IQL_WORKTREE:${PYTHONPATH:-}"

cd "$IQL_WORKTREE"
exec /home/md666/miniforge3/bin/conda run -n dif torchrun \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  ddpo/main_diffsbdd_offline_iql_ddp.py \
  --config "$CONFIG_PATH" \
  --data-root "$PROJECT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --log-level INFO
