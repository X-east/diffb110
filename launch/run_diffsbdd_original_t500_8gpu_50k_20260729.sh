#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
SNAPSHOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?Usage: $0 RUN_DIR}"
CHECKPOINT="${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
PDBFILE="${ROOT}/data/8GJL.pdb"
WORKER="${SNAPSHOT_DIR}/run_diffsbdd_50k_rank.py"
SUMMARIZER="${SNAPSHOT_DIR}/summarize_diffsbdd_50k_run.py"
AGGREGATOR="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/scripts/aggregate_sdf.py"

NUM_RANKS=8
SHARDS_PER_RANK=625
ATTEMPTS_PER_SHARD=10
BATCH_SIZE=10
TIMESTEPS=500

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/ranks" "${RUN_DIR}/aggregated"
echo "$(date -Is) INFO launch target=8GJL/C1101 attempts=50000 timesteps=${TIMESTEPS} seeds=1-8" | tee -a "${RUN_DIR}/logs/launch.log"

pids=()
for rank in $(seq 0 7); do
  seed=$((rank + 1))
  rank_dir="${RUN_DIR}/ranks/rank_$(printf '%02d' "${rank}")"
  mkdir -p "${rank_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${rank}"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    conda run -n dif python "${WORKER}" \
      --repo-root "${ROOT}" \
      --checkpoint "${CHECKPOINT}" \
      --pdbfile "${PDBFILE}" \
      --ref-ligand "C:1101" \
      --output-dir "${rank_dir}" \
      --rank "${rank}" \
      --seed "${seed}" \
      --num-shards "${SHARDS_PER_RANK}" \
      --attempts-per-shard "${ATTEMPTS_PER_SHARD}" \
      --batch-size "${BATCH_SIZE}" \
      --timesteps "${TIMESTEPS}" \
      --reconstruction-method openbabel \
      --resume
  ) > "${RUN_DIR}/logs/rank_$(printf '%02d' "${rank}").log" 2>&1 &
  pids+=("$!")
  echo "$(date -Is) INFO rank=${rank} gpu=${rank} seed=${seed} pid=${pids[-1]}" | tee -a "${RUN_DIR}/logs/launch.log"
done

failed=0
for rank in $(seq 0 7); do
  if wait "${pids[${rank}]}"; then
    echo "$(date -Is) INFO rank=${rank} completed" | tee -a "${RUN_DIR}/logs/launch.log"
  else
    echo "$(date -Is) ERROR rank=${rank} failed log=${RUN_DIR}/logs/rank_$(printf '%02d' "${rank}").log" | tee -a "${RUN_DIR}/logs/launch.log"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "$(date -Is) ERROR run incomplete" | tee -a "${RUN_DIR}/logs/launch.log"
  exit 1
fi

conda run -n dif python "${AGGREGATOR}" \
  --model DiffSBDD \
  --output "${RUN_DIR}/aggregated/diffsbdd_50k.sdf" \
  --limit 50000 \
  "${RUN_DIR}/ranks" \
  > "${RUN_DIR}/logs/aggregate.log" 2>&1

conda run -n dif python "${SUMMARIZER}" \
  --run-dir "${RUN_DIR}" \
  --expected-ranks "${NUM_RANKS}" \
  --expected-shards-per-rank "${SHARDS_PER_RANK}" \
  --attempts-per-shard "${ATTEMPTS_PER_SHARD}" \
  --timesteps "${TIMESTEPS}" \
  > "${RUN_DIR}/logs/summary.log" 2>&1

echo "$(date -Is) INFO run complete summary=${RUN_DIR}/summary.json" | tee -a "${RUN_DIR}/logs/launch.log"
