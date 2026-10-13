#!/usr/bin/env bash
# DePPA 8GJL replica — 方案B DDP (8 GPUs).
# 全局 n_samples=64 按 world_size=8 切到每卡 8；跨卡 gather raw reward 分量，
# 全局算 advantage（rank-normal + 加权 + µB-3σB + z-score）以复刻原 n_samples=64
# 统计语义。这是探索性改造（DePPA 官方无多卡路径），非原论文训练入口。
# 论文超参: rollouts=100, timesteps=500, inference_interval=10, ppo_batch_size=32,
# lr=1e-5, clip=0.2, grad_norm=0.5, QuickVina2. 权重 QED/SA/Vina/Dist=0.2/0.2/0.5/0.1.
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
HERE="${ROOT}/analysis/deppa_8gjl_full_replica_20260720"
DEPPA_DIR="${ROOT}/reference project/代码/DePPA"

# 可覆盖变量（smoke: ROLL=2 SAMPLES=64 RW=2 bash run_full_replica_ddp.sh）
ROLL=${ROLL:-100}
SAMPLES=${SAMPLES:-64}
RW=${RW:-4}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
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
LOG="${HERE}/output/train_ddp_${TS}.log"
mkdir -p "${HERE}/output"

cd "${DEPPA_DIR}"

echo "[replica-ddp] start $(date '+%F %T')" | tee -a "${LOG}"
echo "[replica-ddp] env=deppa_ref  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  n_samples_global=${SAMPLES}  rollouts=${ROLL}" | tee -a "${LOG}"
echo "[replica-ddp] qvina2=$(command -v qvina2)" | tee -a "${LOG}"
echo "[replica-ddp] log=${LOG}" | tee -a "${LOG}"

/home/md666/miniforge3/envs/deppa_ref/bin/python -u -m torch.distributed.run --nproc_per_node=8 --master_port=29500 \
  batch_generate_ligands_rl_ddp.py \
  "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
  --dataset_dir "${HERE}/target" \
  --output_dir "${HERE}/output_ddp" \
  --limit 0:1 \
  --sanitize \
  --mode_num_nodes_lig sample \
  --n_samples "${SAMPLES}" \
  --rollouts "${ROLL}" \
  --timesteps 500 \
  --inference_interval 10 \
  --ppo_batch_size 32 \
  --reward_workers "${RW}" \
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

echo "[replica-ddp] done $(date '+%F %T')" | tee -a "${LOG}"
