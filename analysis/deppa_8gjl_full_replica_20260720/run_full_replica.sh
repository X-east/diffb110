#!/usr/bin/env bash
# DePPA 8GJL full replica (single GPU).
# Paper hyper-parameters: rollouts=100, timesteps=500, inference_interval=10,
# ppo_batch_size=32, lr=1e-5, clip=0.2, grad_norm=0.5, QuickVina2 docking.
# Reward weights per user choice: QED/SA/Vina/Dist = 0.2/0.2/0.5/0.1 (paper
# main text). n_samples reduced 64 -> 16 and ppo_batch_size 32 -> 16 due to the
# 10GB RTX 3080 memory limit: paper 64 OOMs at the sampling peak (~8.7GiB) and
# 32 OOMs during PPO (episode_length=50 observation storage + large 8GJL pocket
# EGNN activations, ~9.2GiB). n_samples/ppo_batch_size only affect per-rollout
# sample count, not algorithm correctness (rule 18.3: batch/sample count is
# memory-constrained and customizable). Persistent optimizer + per-rollout
# checkpoint save via the patched batch_generate_ligands_rl.py.
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
HERE="${ROOT}/analysis/deppa_8gjl_full_replica_20260720"
DEPPA_DIR="${ROOT}/reference project/代码/DePPA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_THREADS=4
export PYTHONPATH="${DEPPA_DIR}:${PYTHONPATH:-}"
# QuickVina2 lives in the diffgui env bin; deppa_ref uses it via PATH.
export PATH="/home/md666/miniforge3/envs/diffgui/bin:${PATH}"
export PYTHONHASHSEED=1
# 10GB 3080: reduce fragmentation so the sampling peak fits.
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${HERE}/output/train_${TS}.log"
mkdir -p "${HERE}/output"

cd "${DEPPA_DIR}"

echo "[replica] start $(date '+%F %T')" | tee -a "${LOG}"
echo "[replica] env=deppa_ref  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"
echo "[replica] qvina2=$(command -v qvina2)" | tee -a "${LOG}"
echo "[replica] log=${LOG}" | tee -a "${LOG}"

taskset -c 8-15 /home/md666/miniforge3/envs/deppa_ref/bin/python -u batch_generate_ligands_rl.py \
  "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
  --dataset_dir "${HERE}/target" \
  --output_dir "${HERE}/output" \
  --limit 0:1 \
  --sanitize \
  --mode_num_nodes_lig sample \
  --n_samples 16 \
  --rollouts 100 \
  --timesteps 500 \
  --inference_interval 10 \
  --ppo_batch_size 16 \
  --reward_workers 4 \
  --ppo_lr 1e-5 \
  --clip_range 0.2 \
  --max_grad_norm 0.5 \
  --kl_coeff_pretrain 0.0 \
  --w_qed 0.2 \
  --w_sa 0.2 \
  --w_vina_score 0.5 \
  --w_distance 0.1 \
  --w_strain 0.0 \
  --vina_center_xyz_mode ref \
  --wandb_mode disabled \
  --seed 1 \
  --skip_final_eval 2>&1 | tee -a "${LOG}"

echo "[replica] done $(date '+%F %T')" | tee -a "${LOG}"
