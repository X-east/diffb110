#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP="${SCRIPT_DIR}"
INPUTS="${EXP}/inputs"
SCRIPTS="${ROOT}/analysis/8gjl_c1101_benchmark_20260603_500mol_500steps/scripts"

TOTAL="${TOTAL:-500}"
TIMESTEPS="${TIMESTEPS:-500}"
GPUS_STR="${GPUS:-0 1 2 3 4 5 6}"
IFS=' ' read -r -a GPUS <<< "${GPUS_STR}"

mkdir -p "${EXP}/logs" "${EXP}/outputs" "${EXP}/aggregated" "${EXP}/evaluation" "${EXP}/configs/diffgui"

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

join_by_comma() {
  local IFS=","
  echo "$*"
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

run_diffsbdd_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/diffsbdd/gpu_${gpu_id}"
  mkdir -p "${out_dir}"
  /usr/bin/time -p conda run -n dif python "${SCRIPTS}/run_diffsbdd_shard.py" \
    --repo-root "${ROOT}" \
    --checkpoint "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
    --pdbfile "${ROOT}/data/5xzr.pdb" \
    --ref-ligand "A:601" \
    --outfile "${out_dir}/generated.sdf" \
    --n-samples "${count}" \
    --batch-size 10 \
    --timesteps "${TIMESTEPS}" \
    --seed "$((202606110 + gpu_id * 1009))" \
    --reconstruction-method openbabel
}

aggregate_diffsbdd() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffSBDD \
    --output "${EXP}/aggregated/diffsbdd.sdf" \
    --limit "${TOTAL}" \
    "${EXP}/outputs/diffsbdd"
}

prepare_diffgui_configs() {
  conda run -n dif python "${SCRIPTS}/prepare_diffgui_configs.py" \
    --out-dir "${EXP}/configs/diffgui" \
    --target "${INPUTS}/5XZR_8J6_pocket8A.pdb" \
    --counts "$(join_by_comma "${COUNTS[@]}")" \
    --base-seed 20260611
}

run_diffgui_one() {
  local gpu_id="$1"
  local count="$2"
  local project="${ROOT}/reference project/代码/DiffGui"
  local out_dir="${EXP}/outputs/diffgui/gpu_${gpu_id}"
  local cfg="${EXP}/configs/diffgui/diffgui_c1101_gpu${gpu_id}.yml"
  local mpl_dir="${EXP}/mplconfig/diffgui/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  cd "${project}"
  export DIFFGUI_NUM_TIMESTEPS="${TIMESTEPS}"
  export PYTHONPATH="${ROOT}/analysis/8gjl_c1101_benchmark_20260603_500mol_500steps/runtime_patch/diffgui:${project}${PYTHONPATH:+:${PYTHONPATH}}"
  export MPLCONFIGDIR="${mpl_dir}"
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  /usr/bin/time -p conda run -n diffgui python scripts/sample.py \
    --config "${cfg}" \
    --outdir "${out_dir}" \
    --device cuda:0 \
    --batch_size 2
}

aggregate_diffgui() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffGui \
    --output "${EXP}/aggregated/diffgui.sdf" \
    --limit "${TOTAL}" \
    "${EXP}/outputs/diffgui"
}

run_deppa_one() {
  local gpu_id="$1"
  local count="$2"
  local project="${ROOT}/reference project/代码/DePPA"
  local out_dir="${EXP}/outputs/deppa/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/deppa/gpu_${gpu_id}"
  mkdir -p "${EXP}/outputs/deppa" "${mpl_dir}"
  export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n deppa_ref python "${project}/test.py" \
    "${ROOT}/analysis/reference_project_generation_20260530_173617/deppa_downloads/crossdocked_fullatom_cond.ckpt" \
    --test_dir "${INPUTS}/deppa_5xzr_test_pocket" \
    --outdir "${out_dir}" \
    --n_samples "${count}" \
    --batch_size 10 \
    --timesteps "${TIMESTEPS}"
}

aggregate_deppa() {
  local processed_dirs=()
  local gpu_id
  for gpu_id in "${GPUS[@]}"; do
    processed_dirs+=("${EXP}/outputs/deppa/gpu_${gpu_id}/processed")
  done
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DePPA \
    --output "${EXP}/aggregated/deppa.sdf" \
    --limit "${TOTAL}" \
    "${processed_dirs[@]}"
}

evaluate_core() {
  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${EXP}/evaluation/core" \
    --sdf "DiffSBDD=${EXP}/aggregated/diffsbdd.sdf" \
    --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" \
    --sdf "DePPA=${EXP}/aggregated/deppa.sdf"
}

build_counts "${TOTAL}"
echo "[5xzr_core] started at $(date -Is)"
echo "[5xzr_core] target=5XZR/8J6_A_601 total=${TOTAL} timesteps=${TIMESTEPS} gpus=${GPUS[*]} counts=${COUNTS[*]}"

prepare_diffgui_configs

run_parallel_shards diffsbdd run_diffsbdd_one
aggregate_diffsbdd

run_parallel_shards diffgui run_diffgui_one
aggregate_diffgui

run_parallel_shards deppa run_deppa_one
aggregate_deppa

evaluate_core

echo "[5xzr_core] finished at $(date -Is)"
