#!/usr/bin/env bash
# Formal DePPA correctness smoke: single GPU, batch=8, 2 rollouts, 500 steps.
# Exercises the integrated batch_generate_ligands_rl.py (persistent optimizer +
# full checkpoint save). Run from this directory.
set -euo pipefail

ROOT="/data/disk6/DiffB110"
HERE="${ROOT}/analysis/deppa_tests"
DEppa_DIR="${ROOT}/reference project/代码/DePPA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_THREADS=4
export PYTHONPATH="${DEppa_DIR}:${PYTHONPATH:-}"
export PATH="/home/md666/miniforge3/envs/diffgui/bin:${PATH}"
export PYTHONHASHSEED=1

cd "${DEppa_DIR}"

taskset -c 8-11 /home/md666/miniforge3/envs/deppa_ref/bin/python -u batch_generate_ligands_rl.py \
  "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
  --dataset_dir "${HERE}/target" \
  --output_dir "${HERE}/correctness_smoke/output" \
  --limit 0:1 \
  --sanitize \
  --mode_num_nodes_lig sample \
  --n_samples 8 \
  --rollouts 2 \
  --timesteps 500 \
  --inference_interval 10 \
  --ppo_batch_size 32 \
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
  --skip_final_eval
