#!/usr/bin/env bash
# Dock the existing T=200 and T=500 DiffGui step-sweep batches with the very same
# UniDock configuration used by the five-model benchmark (exhaustiveness=8,
# scoring vina, search_mode fast, box 20 A at 207.293/212.430/172.771).
set -uo pipefail
ROOT=/home/md666/data/disk6/DiffB110
SRC="${ROOT}/analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2"
EXP="${ROOT}/analysis/benchmark/benchmark5/diffgui_200steps_validity_260918"
DOCK="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/dock_sdf_unidock.py"

REPS=(rep1_seed1_7 rep2_seed8_14 rep3_seed15_21)

run_one() {
  local tag="$1" gpu="$2" sdf="$3"
  conda run -n dif python "${DOCK}" \
    --model "DiffGui_${tag}" --input-sdf "${sdf}" \
    --receptor "${ROOT}/data/8gjl.pdbqt" \
    --out-dir "${EXP}/docking/${tag}" \
    --unidock-bin /opt/unidock-1.1.3/bin/unidock \
    --gpu-id "${gpu}" --batch-size 128 --exhaustiveness 8 \
    >"${EXP}/logs/dock_${tag}.log" 2>&1
  echo "[$(date -Is)] docked ${tag} exit=$?"
}

idx=0
for step in 200 500; do
  for rep in "${REPS[@]}"; do
    tag="${step}steps_${rep}"
    sdf="${SRC}/${tag}/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    run_one "${tag}" "$((idx % 5))" "${sdf}" &
    idx=$((idx + 1))
  done
done
wait
echo "all six dockings finished"
