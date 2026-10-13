#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/disk6/DiffB110"
RUN_DIR="${ROOT}/analysis/deppa_8gjl_single_gpu_smoke_20260720"
DEppa_DIR="${ROOT}/reference project/代码/DePPA"

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR="${RUN_DIR}/mplconfig"
export PYTHONPATH="${DEppa_DIR}:${PYTHONPATH:-}"
export PYTHONHASHSEED=1
export PATH="/home/md666/miniforge3/envs/diffgui/bin:${PATH}"

taskset -c 8-11 /home/md666/miniforge3/envs/deppa_ref/bin/python -u \
  "${RUN_DIR}/smoke_batch_generate_ligands_rl.py" \
  "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
  --dataset_dir "${RUN_DIR}/target" \
  --output_dir "${RUN_DIR}/output" \
  --limit 0:1 \
  --sanitize \
  --mode_num_nodes_lig sample \
  --n_samples 4 \
  --rollouts 1 \
  --timesteps 100 \
  --inference_interval 5 \
  --ppo_batch_size 32 \
  --reward_workers 4 \
  --ppo_lr 1e-5 \
  --clip_range 0.2 \
  --kl_coeff_pretrain 0.0 \
  --w_qed 0.2 \
  --w_sa 0.2 \
  --w_vina_score 0.5 \
  --w_distance 0.1 \
  --w_strain 0.0 \
  --vina_center_xyz_mode ref \
  --wandb_mode disabled \
  --seed 1 \
  --skip_final_eval
