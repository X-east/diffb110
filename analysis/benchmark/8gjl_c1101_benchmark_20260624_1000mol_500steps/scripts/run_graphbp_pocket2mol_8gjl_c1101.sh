#!/usr/bin/env bash
# 8GJL/C1101 baseline generation for GraphBP and Pocket2Mol.
# Full mode follows the original benchmark shape: 7 GPUs, 1000 attempts total.
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="/data/disk6/DiffB110"
EXP="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps"
INPUTS="${EXP}/inputs"
SCRIPTS="${EXP}/scripts"
MODE="${MODE:-full}"
CENTER="207.4519,212.5211,172.9314"
TIMESTEPS=500

GRAPHBP_ROOT="${ROOT}/reference project/代码/GraphBP"
GRAPHBP_CHECKPOINT="${GRAPHBP_ROOT}/GraphBP/trained_model/model_33.pth"
P2M_ROOT="${ROOT}/reference project/代码/Pocket2Mol"
P2M_CHECKPOINT="${P2M_ROOT}/ckpt/pretrained_Pocket2Mol.pt"

if [[ "${MODE}" == "full" ]]; then
  GPUS=(1 2 3 4 5 6 7)
  COUNTS=(143 143 143 143 143 143 142)
  RUN_LABEL="full"
  P2M_BEAM_SIZE=50
  P2M_MAX_STEPS=50
elif [[ "${MODE}" == "smoke" ]]; then
  GPUS=(1)
  COUNTS=(2)
  RUN_LABEL="smoke"
  P2M_BEAM_SIZE=20
  P2M_MAX_STEPS=50
else
  echo "MODE must be smoke or full, got: ${MODE}" >&2
  exit 2
fi

P2M_CONFIG_DIR="${EXP}/configs/pocket2mol_graphbp/${RUN_LABEL}"
P2M_OUT_ROOT="${EXP}/outputs/pocket2mol/${RUN_LABEL}"
GRAPHBP_OUT_ROOT="${EXP}/outputs/graphbp/${RUN_LABEL}"
AGG_ROOT="${EXP}/aggregated"
LOG_ROOT="${EXP}/logs/pocket2mol_graphbp"
MASTER_LOG="${LOG_ROOT}/${RUN_LABEL}.master.log"

mkdir -p "${P2M_CONFIG_DIR}" "${P2M_OUT_ROOT}" "${GRAPHBP_OUT_ROOT}" "${AGG_ROOT}" "${LOG_ROOT}"
: >"${MASTER_LOG}"

log() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

run_parallel() {
  local model_name="$1"
  local log_dir="$2"
  shift 2
  mkdir -p "${log_dir}"

  local pids=()
  for idx in "${!GPUS[@]}"; do
    local gpu_id="${GPUS[$idx]}"
    local count="${COUNTS[$idx]}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export PYTHONDONTWRITEBYTECODE=1
      "$@" "${gpu_id}" "${count}"
    ) >"${log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    log "[${model_name}] launched gpu=${gpu_id} count=${count} pid=${pids[-1]}"
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      log "[${model_name}] gpu=${GPUS[$idx]} completed"
    else
      log "[${model_name}] gpu=${GPUS[$idx]} FAILED (see ${log_dir}/gpu_${GPUS[$idx]}.log)"
      failed=1
    fi
  done
  return "${failed}"
}

prepare_pocket2mol_configs() {
  conda run -n dif python "${SCRIPTS}/prepare_pocket2mol_configs.py" \
    --out-dir "${P2M_CONFIG_DIR}" \
    --result-root "${P2M_OUT_ROOT}" \
    --checkpoint "${P2M_CHECKPOINT}" \
    --counts "$(IFS=,; echo "${COUNTS[*]}")" \
    --base-seed 20260625 \
    --beam-size "${P2M_BEAM_SIZE}" \
    --max-steps "${P2M_MAX_STEPS}" | tee -a "${MASTER_LOG}"
}

run_pocket2mol_one() {
  local gpu_id="$1"
  local _count="$2"
  local cfg="${P2M_CONFIG_DIR}/pocket2mol_c1101_gpu${gpu_id}.yml"
  local out_dir="${P2M_OUT_ROOT}/gpu_${gpu_id}"
  mkdir -p "${out_dir}"
  cd "${P2M_ROOT}"
  # conda run does not forward PYTHONPATH; set it outside conda run so that
  # Pocket2Mol's own `utils.*` package is importable.
  PYTHONPATH="${P2M_ROOT}" conda run -n pmdm_modern_ref python sample_for_pdb.py \
    --pdb_path "${INPUTS}/8GJL_C1101_protein_only.pdb" \
    --center "${CENTER}" \
    --bbox_size 23.0 \
    --config "${cfg}" \
    --device cuda \
    --outdir "${out_dir}"
}

run_graphbp_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${GRAPHBP_OUT_ROOT}/gpu_${gpu_id}"
  mkdir -p "${out_dir}"
  conda run -n pmdm_modern_ref python \
    "${SCRIPTS}/run_graphbp_custom_pocket.py" \
    --graphbp-root "${GRAPHBP_ROOT}" \
    --protein "${INPUTS}/8GJL_C1101_protein_only.pdb" \
    --ligand "${INPUTS}/8GJL_C1101_ligand.sdf" \
    --checkpoint "${GRAPHBP_CHECKPOINT}" \
    --output-dir "${out_dir}" \
    --num-samples "${count}" \
    --seed "${gpu_id}" \
    --device cuda
}

aggregate_pocket2mol() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model Pocket2Mol \
    --output "${AGG_ROOT}/pocket2mol.sdf" \
    --limit 1000 \
    "${P2M_OUT_ROOT}" | tee -a "${MASTER_LOG}"
}

aggregate_graphbp() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model GraphBP \
    --output "${AGG_ROOT}/graphbp.sdf" \
    --limit 1000 \
    "${GRAPHBP_OUT_ROOT}" | tee -a "${MASTER_LOG}"
}

main() {
  log "[benchmark] started at $(date -Is)"
  log "[benchmark] mode=${MODE} gpus=${GPUS[*]} counts=${COUNTS[*]} center=${CENTER}"

  log "[benchmark] preparing Pocket2Mol configs"
  prepare_pocket2mol_configs

  log "[benchmark] running Pocket2Mol"
  run_parallel "Pocket2Mol" "${LOG_ROOT}/pocket2mol_${RUN_LABEL}" run_pocket2mol_one
  log "[benchmark] aggregating Pocket2Mol"
  aggregate_pocket2mol

  log "[benchmark] running GraphBP"
  run_parallel "GraphBP" "${LOG_ROOT}/graphbp_${RUN_LABEL}" run_graphbp_one
  log "[benchmark] aggregating GraphBP"
  aggregate_graphbp

  log "[benchmark] completed at $(date -Is)"
}

main "$@"
