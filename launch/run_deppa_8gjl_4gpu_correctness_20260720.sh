#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/disk6/DiffB110"
DEPPA_DIR="${ROOT}/reference project/代码/DePPA"
TARGET_DIR="${ROOT}/analysis/deppa_8gjl_full_replica_20260720/target"
RUN_DIR="${ROOT}/analysis/deppa_8gjl_4gpu_correctness_20260720"

ROLL=${ROLL:-2}
SAMPLES=${SAMPLES:-32}
REWARD_WORKERS=${REWARD_WORKERS:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29501}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONPATH="${DEPPA_DIR}:${PYTHONPATH:-}"
export PATH="/home/md666/miniforge3/envs/diffgui/bin:${PATH}"
export PYTHONHASHSEED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export MPLCONFIGDIR="${RUN_DIR}/mplconfig"

mkdir -p "${RUN_DIR}/output" "${MPLCONFIGDIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="${RUN_DIR}/output/train_${timestamp}.log"

resume_args=()
if [[ -n "${RESUME:-}" ]]; then
  resume_args=(--resume "${RESUME}")
fi

echo "[deppa-4gpu] start=$(date '+%F %T')" | tee -a "${log_path}"
echo "[deppa-4gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} global_samples=${SAMPLES} rollouts=${ROLL}" | tee -a "${log_path}"
echo "[deppa-4gpu] cpu_set=8-23 threads_per_rank=4 qvina2=$(command -v qvina2)" | tee -a "${log_path}"

taskset -c 8-23 /home/md666/miniforge3/envs/deppa_ref/bin/python -u \
  -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${DEPPA_DIR}/batch_generate_ligands_rl_ddp.py" \
  "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
  --dataset_dir "${TARGET_DIR}" \
  --output_dir "${RUN_DIR}/output" \
  --limit 0:1 \
  --sanitize \
  --mode_num_nodes_lig sample \
  --n_samples "${SAMPLES}" \
  --rollouts "${ROLL}" \
  --timesteps 500 \
  --inference_interval 10 \
  --ppo_batch_size 32 \
  --reward_workers "${REWARD_WORKERS}" \
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
  --skip_final_eval \
  "${resume_args[@]}" 2>&1 | tee -a "${log_path}"

echo "[deppa-4gpu] done=$(date '+%F %T')" | tee -a "${log_path}"
