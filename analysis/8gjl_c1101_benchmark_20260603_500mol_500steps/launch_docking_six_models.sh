#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
EXP="${ROOT}/analysis/8gjl_c1101_benchmark_20260603_500mol_500steps"
SCRIPT="${EXP}/scripts/dock_sdf_unidock.py"
RECEPTOR="${ROOT}/data/8gjl.pdbqt"
OUT_ROOT="${EXP}/docking/unidock_fast_8gjl_c1101"

mkdir -p "${OUT_ROOT}/logs"

run_one() {
  local model="$1"
  local sdf="$2"
  local gpu_id="$3"
  local out_dir="${OUT_ROOT}/${model}"
  local mpl_dir="${EXP}/mplconfig/docking/${model}"
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
    --batch-size 128 \
    --center-x 207.293 \
    --center-y 212.430 \
    --center-z 172.771 \
    --size-x 20.0 \
    --size-y 20.0 \
    --size-z 20.0 \
    --search-mode fast \
    --scoring vina \
    --num-modes 1 \
    --seed 42
}

echo "[docking] started at $(date -Is)"

pids=()
run_one DiffSBDD "${EXP}/aggregated/diffsbdd.sdf" 0 > "${OUT_ROOT}/logs/DiffSBDD.log" 2>&1 & pids+=("$!")
run_one DiffGui "${EXP}/aggregated/diffgui.sdf" 1 > "${OUT_ROOT}/logs/DiffGui.log" 2>&1 & pids+=("$!")
run_one DePPA "${EXP}/aggregated/deppa.sdf" 2 > "${OUT_ROOT}/logs/DePPA.log" 2>&1 & pids+=("$!")
run_one PMDM "${EXP}/aggregated/pmdm.sdf" 3 > "${OUT_ROOT}/logs/PMDM.log" 2>&1 & pids+=("$!")
run_one DiffInt "${EXP}/aggregated/diffint.sdf" 4 > "${OUT_ROOT}/logs/DiffInt.log" 2>&1 & pids+=("$!")
run_one DiffPharma "${EXP}/aggregated/diffpharma.sdf" 5 > "${OUT_ROOT}/logs/DiffPharma.log" 2>&1 & pids+=("$!")

failed=0
models=(DiffSBDD DiffGui DePPA PMDM DiffInt DiffPharma)
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

echo "[docking] finished at $(date -Is)"
