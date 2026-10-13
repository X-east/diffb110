#!/usr/bin/env bash
# Sequentially run Pocket2Mol+GraphBP, then TargetDiff with seed=gpu_id.
# Pocket2Mol+GraphBP run first on GPU 1-7, then TargetDiff on GPU 1-7.
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
EXP="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps"
SCRIPTS="${EXP}/scripts"

LOG="${EXP}/logs/sequential_benchmark_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${EXP}/logs"
: >"${LOG}"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

log "=== Stage 1: GraphBP + Pocket2Mol (seed=gpu_id) ==="
MODE=full bash "${SCRIPTS}/run_graphbp_pocket2mol_8gjl_c1101.sh" 2>&1 | tee -a "${LOG}"
log "=== Stage 1 completed ==="

log "=== Stage 2: TargetDiff rerun (seed=gpu_id) ==="
# TargetDiff configs already have seed=gpu_id (1-7) overwritten in-place.
# Run via run_gcdm_targetdiff_full.sh but only the targetdiff part.
COUNTS=(143 143 143 143 143 143 142)
GPUS=(1 2 3 4 5 6 7)
TD_LOG_DIR="${EXP}/logs/targetdiff_full"
mkdir -p "${TD_LOG_DIR}"

pids=()
for gpu_id in "${GPUS[@]}"; do
  local_count="${COUNTS[$((gpu_id - 1))]}"
  local_cfg="${EXP}/configs/targetdiff_full/targetdiff_c1101_gpu${gpu_id}.yml"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTHONDONTWRITEBYTECODE=1
    cd "/home/md666/data/disk6/XDC/targetdiff"
    conda run -n targetdiff python scripts/sample_from_protein_ligand.py "${local_cfg}"
  ) >"${TD_LOG_DIR}/gpu_${gpu_id}.log" 2>&1 &
  pids+=("$!")
  log "[targetdiff] launched gpu=${gpu_id} count=${local_count} pid=${pids[-1]}"
done

failed=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    log "[targetdiff] gpu=${GPUS[$idx]} completed"
  else
    log "[targetdiff] gpu=${GPUS[$idx]} FAILED"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  log "[targetdiff] one or more shards failed"
  exit 1
fi

log "=== Stage 2: aggregating TargetDiff ==="
conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
  --model TargetDiff \
  --output "${EXP}/aggregated/targetdiff.sdf" \
  --limit 1000 \
  "${EXP}/outputs/targetdiff" | tee -a "${LOG}"

log "=== All stages completed ==="
