#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
EXP="${ROOT}/analysis/8gjl_c1101_benchmark_20260603_500mol_500steps"
INPUTS="${EXP}/inputs"
SCRIPTS="${EXP}/scripts"
COUNTS=(72 72 72 71 71 71 71)
GPUS=(0 1 2 3 4 5 6)
TIMESTEPS=500

mkdir -p "${EXP}/logs" "${EXP}/outputs" "${EXP}/aggregated"

run_parallel_shards() {
  local model_name="$1"
  local log_dir="${EXP}/logs/${model_name}"
  mkdir -p "${log_dir}"
  shift

  local pids=()
  local gpu_id
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[${gpu_id}]}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export PYTHONDONTWRITEBYTECODE=1
      "$@" "${gpu_id}" "${count}"
    ) > "${log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    echo "[${model_name}] launched gpu=${gpu_id} count=${count} pid=${pids[-1]}"
  done

  local failed=0
  local idx
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[${model_name}] gpu=${idx} completed"
    else
      echo "[${model_name}] gpu=${idx} failed; see ${log_dir}/gpu_${idx}.log"
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
    --pdbfile "${ROOT}/data/8GJL.pdb" \
    --ref-ligand "C:1101" \
    --outfile "${out_dir}/generated.sdf" \
    --n-samples "${count}" \
    --batch-size 10 \
    --timesteps "${TIMESTEPS}" \
    --seed "$((202606030 + gpu_id * 1009))" \
    --reconstruction-method openbabel
}

aggregate_diffsbdd() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffSBDD \
    --output "${EXP}/aggregated/diffsbdd.sdf" \
    --limit 500 \
    "${EXP}/outputs/diffsbdd"
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
  export PYTHONPATH="${EXP}/runtime_patch/diffgui:${project}${PYTHONPATH:+:${PYTHONPATH}}"
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
    --limit 500 \
    "${EXP}/outputs/diffgui"
}

run_deppa_one() {
  local gpu_id="$1"
  local count="$2"
  local project="${ROOT}/reference project/代码/DePPA"
  local out_dir="${EXP}/outputs/deppa_rerun/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/deppa/gpu_${gpu_id}"
  mkdir -p "${EXP}/outputs/deppa_rerun" "${mpl_dir}"
  export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n deppa_ref python "${project}/test.py" \
    "${ROOT}/analysis/reference_project_generation_20260530_173617/deppa_downloads/crossdocked_fullatom_cond.ckpt" \
    --test_dir "${INPUTS}/deppa_test_pocket" \
    --outdir "${out_dir}" \
    --n_samples "${count}" \
    --batch_size 10 \
    --timesteps "${TIMESTEPS}"
}

aggregate_deppa() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DePPA \
    --output "${EXP}/aggregated/deppa.sdf" \
    --limit 500 \
    "${EXP}/outputs/deppa_rerun"
}

evaluate_core() {
  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${EXP}/evaluation/core" \
    --sdf "DiffSBDD=${EXP}/aggregated/diffsbdd.sdf" \
    --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" \
    --sdf "DePPA=${EXP}/aggregated/deppa.sdf"
}

echo "[benchmark] core benchmark started at $(date -Is)"
echo "[benchmark] target=8GJL/C1101 timesteps=${TIMESTEPS} total_per_model=500 gpus=${GPUS[*]}"

run_parallel_shards diffsbdd run_diffsbdd_one
aggregate_diffsbdd

run_parallel_shards diffgui run_diffgui_one
aggregate_diffgui

run_parallel_shards deppa run_deppa_one
aggregate_deppa

evaluate_core

echo "[benchmark] core benchmark finished at $(date -Is)"
