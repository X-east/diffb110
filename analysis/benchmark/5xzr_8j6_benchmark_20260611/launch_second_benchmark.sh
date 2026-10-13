#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP="${SCRIPT_DIR}"
INPUTS="${EXP}/inputs"
SCRIPTS="${ROOT}/analysis/8gjl_c1101_benchmark_20260603_500mol_500steps/scripts"
INTERACTION_LIGAND_SDF="${INPUTS}/5XZR_8J6_ligand_rdkit_fixed.sdf"

TOTAL="${TOTAL:-500}"
TIMESTEPS="${TIMESTEPS:-500}"
GPUS_STR="${GPUS:-0 1 2 3 4 5 6}"
PMDM_NUM_ATOM="${PMDM_NUM_ATOM:-28}"
IFS=' ' read -r -a GPUS <<< "${GPUS_STR}"

mkdir -p "${EXP}/logs" "${EXP}/outputs" "${EXP}/aggregated" "${EXP}/evaluation" "${EXP}/work"

build_counts() {
  local total="$1"
  local n_gpu="${#GPUS[@]}"
  local base=$((total / n_gpu))
  local rem=$((total % n_gpu))
  COUNTS=()
  local idx
  for idx in "${!GPUS[@]}"; do
    local count="${base}"
    if [[ "${idx}" -lt "${rem}" ]]; then
      count=$((count + 1))
    fi
    COUNTS+=("${count}")
  done
}

run_parallel_shards() {
  local model_name="$1"
  local log_dir="${EXP}/logs/${model_name}"
  mkdir -p "${log_dir}"
  shift

  local pids=()
  local idx
  for idx in "${!GPUS[@]}"; do
    local gpu_id="${GPUS[${idx}]}"
    local count="${COUNTS[${idx}]}"
    if [[ "${count}" -le 0 ]]; then
      echo "[${model_name}] skip gpu=${gpu_id} count=0"
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export PYTHONDONTWRITEBYTECODE=1
      "$@" "${gpu_id}" "${count}"
    ) > "${log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    echo "[${model_name}] launched gpu=${gpu_id} count=${count} pid=${pids[-1]}"
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[${model_name}] shard_index=${idx} completed"
    else
      echo "[${model_name}] shard_index=${idx} failed; see ${log_dir}"
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi
}

run_pmdm_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/pmdm/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/pmdm/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n pmdm_modern_ref python "${ROOT}/analysis/modern_sbdd_20260531/run_pmdm_for_pdb.py" \
    --repo "${ROOT}/reference project/代码/modern_sbdd_20260531/PMDM" \
    --ckpt "${ROOT}/analysis/modern_sbdd_20260531/assets/PMDM/500.pt" \
    --pdb-path "${INPUTS}/5XZR_8J6_pocket8A.pdb" \
    --outdir "${out_dir}" \
    --num-atom "${PMDM_NUM_ATOM}" \
    --num-samples "${count}" \
    --batch-size 32 \
    --n-steps "${TIMESTEPS}" \
    --device cuda
}

aggregate_pmdm() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model PMDM \
    --output "${EXP}/aggregated/pmdm.sdf" \
    --limit "${TOTAL}" \
    "${EXP}/outputs/pmdm"
}

run_diffint_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/diffint/gpu_${gpu_id}"
  local work_dir="${EXP}/work/diffint/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/diffint/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${work_dir}" "${mpl_dir}"
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n diffint_modern_ref python "${SCRIPTS}/run_interaction_model_shard.py" \
    --kind diffint \
    --repo "${ROOT}/reference project/代码/modern_sbdd_20260531/DiffInt" \
    --checkpoint "${ROOT}/reference project/代码/modern_sbdd_20260531/DiffInt/checkpoints/best_model.ckpt" \
    --pdb "${INPUTS}/5XZR_8J6_protein_only.pdb" \
    --sdf "${INTERACTION_LIGAND_SDF}" \
    --outdir "${out_dir}" \
    --workdir "${work_dir}" \
    --n-samples "${count}" \
    --batch-size 72 \
    --timesteps "${TIMESTEPS}"
}

aggregate_diffint() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffInt \
    --output "${EXP}/aggregated/diffint.sdf" \
    --limit "${TOTAL}" \
    "${EXP}/outputs/diffint/gpu_0/processed" \
    "${EXP}/outputs/diffint/gpu_1/processed" \
    "${EXP}/outputs/diffint/gpu_2/processed" \
    "${EXP}/outputs/diffint/gpu_3/processed" \
    "${EXP}/outputs/diffint/gpu_4/processed" \
    "${EXP}/outputs/diffint/gpu_5/processed" \
    "${EXP}/outputs/diffint/gpu_6/processed"
}

run_diffpharma_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/diffpharma/gpu_${gpu_id}"
  local work_dir="${EXP}/work/diffpharma/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/diffpharma/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${work_dir}" "${mpl_dir}"
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n diffpharma_modern_ref python "${SCRIPTS}/run_interaction_model_shard.py" \
    --kind diffpharma \
    --repo "${ROOT}/reference project/代码/modern_sbdd_20260531/DiffPharma" \
    --checkpoint "${ROOT}/analysis/modern_sbdd_20260531/assets/DiffPharma/DiffPharma_best.ckpt" \
    --pdb "${INPUTS}/5XZR_8J6_protein_only.pdb" \
    --sdf "${INTERACTION_LIGAND_SDF}" \
    --outdir "${out_dir}" \
    --workdir "${work_dir}" \
    --n-samples "${count}" \
    --batch-size 24 \
    --timesteps "${TIMESTEPS}"
}

aggregate_diffpharma() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffPharma \
    --output "${EXP}/aggregated/diffpharma.sdf" \
    --limit "${TOTAL}" \
    "${EXP}/outputs/diffpharma/gpu_0/processed" \
    "${EXP}/outputs/diffpharma/gpu_1/processed" \
    "${EXP}/outputs/diffpharma/gpu_2/processed" \
    "${EXP}/outputs/diffpharma/gpu_3/processed" \
    "${EXP}/outputs/diffpharma/gpu_4/processed" \
    "${EXP}/outputs/diffpharma/gpu_5/processed" \
    "${EXP}/outputs/diffpharma/gpu_6/processed"
}

evaluate_all_available() {
  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${EXP}/evaluation/six_models" \
    --sdf "DiffSBDD=${EXP}/aggregated/diffsbdd.sdf" \
    --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" \
    --sdf "DePPA=${EXP}/aggregated/deppa.sdf" \
    --sdf "PMDM=${EXP}/aggregated/pmdm.sdf" \
    --sdf "DiffInt=${EXP}/aggregated/diffint.sdf" \
    --sdf "DiffPharma=${EXP}/aggregated/diffpharma.sdf"
}

build_counts "${TOTAL}"
echo "[5xzr_second] started at $(date -Is)"
echo "[5xzr_second] target=5XZR/8J6_A_601 total=${TOTAL} timesteps=${TIMESTEPS} gpus=${GPUS[*]} counts=${COUNTS[*]} pmdm_num_atom=${PMDM_NUM_ATOM}"

run_parallel_shards pmdm run_pmdm_one
aggregate_pmdm

run_parallel_shards diffint run_diffint_one
aggregate_diffint

run_parallel_shards diffpharma run_diffpharma_one
aggregate_diffpharma

evaluate_all_available

echo "[5xzr_second] finished at $(date -Is)"
