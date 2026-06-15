#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP="${SCRIPT_DIR}"
SCRIPT="${ROOT}/analysis/8gjl_c1101_benchmark_20260603_500mol_500steps/scripts/dock_sdf_unidock.py"
RECEPTOR="${EXP}/inputs/5XZR_8J6_pocket8A.pdbqt"
OUT_ROOT="${EXP}/docking/unidock_fast_5xzr_8j6"

CENTER_X="${CENTER_X:--21.484}"
CENTER_Y="${CENTER_Y:-70.592}"
CENTER_Z="${CENTER_Z:-21.081}"
SIZE_X="${SIZE_X:-20.0}"
SIZE_Y="${SIZE_Y:-20.0}"
SIZE_Z="${SIZE_Z:-20.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"

mkdir -p "${OUT_ROOT}/logs"

run_one() {
  local model="$1"
  local sdf="$2"
  local gpu_id="$3"
  local out_dir="${OUT_ROOT}/${model}"
  local mpl_dir="${EXP}/mplconfig/docking/${model}"
  if [[ ! -s "${sdf}" ]]; then
    echo "[docking] skip ${model}: missing ${sdf}"
    return 0
  fi
  mkdir -p "${out_dir}" "${mpl_dir}"
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  export PYTHONDONTWRITEBYTECODE=1
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n dif python "${SCRIPT}" \
    --model "${model}" \
    --input-sdf "${sdf}" \
    --receptor "${RECEPTOR}" \
    --out-dir "${out_dir}" \
    --gpu-id 0 \
    --batch-size "${BATCH_SIZE}" \
    --center-x "${CENTER_X}" \
    --center-y "${CENTER_Y}" \
    --center-z "${CENTER_Z}" \
    --size-x "${SIZE_X}" \
    --size-y "${SIZE_Y}" \
    --size-z "${SIZE_Z}" \
    --search-mode fast \
    --scoring vina \
    --num-modes 1 \
    --seed 42
}

echo "[5xzr_docking] started at $(date -Is)"
echo "[5xzr_docking] receptor=${RECEPTOR} center=${CENTER_X},${CENTER_Y},${CENTER_Z} size=${SIZE_X},${SIZE_Y},${SIZE_Z}"

pids=()
models=()
launch_model() {
  local model="$1"
  local sdf="$2"
  local gpu="$3"
  run_one "${model}" "${sdf}" "${gpu}" > "${OUT_ROOT}/logs/${model}.log" 2>&1 &
  pids+=("$!")
  models+=("${model}")
  echo "[docking] launched ${model} gpu=${gpu} pid=${pids[-1]}"
}

launch_model Native8J6 "${EXP}/inputs/5XZR_8J6_ligand.sdf" 0
launch_model DiffSBDD "${EXP}/aggregated/diffsbdd.sdf" 0
launch_model DiffGui "${EXP}/aggregated/diffgui.sdf" 1
launch_model DePPA "${EXP}/aggregated/deppa.sdf" 2
launch_model PMDM "${EXP}/aggregated/pmdm.sdf" 3
launch_model DiffInt "${EXP}/aggregated/diffint.sdf" 4
launch_model DiffPharma "${EXP}/aggregated/diffpharma.sdf" 5

failed=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[docking] ${models[$idx]} completed"
  else
    echo "[docking] ${models[$idx]} failed; see ${OUT_ROOT}/logs/${models[$idx]}.log"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

echo "[5xzr_docking] finished at $(date -Is)"
