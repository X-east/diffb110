#!/usr/bin/env bash
# Same DiffGui molecules, two UniDock exhaustiveness settings.
# Everything else is held at the 500-step benchmark configuration
# (receptor data/8gjl.pdbqt, box center 207.293/212.430/172.771, size 20 A,
#  scoring vina, search_mode fast, batch 128).
set -uo pipefail
ROOT=/home/md666/data/disk6/DiffB110
E="${ROOT}/analysis/benchmark/benchmark5/five_model_500steps_full_eval_260903/docking_exh_scan"
S="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/dock_sdf_unidock.py"
I="${ROOT}/analysis/benchmark/benchmark5/five_model_repeats_500steps_260826/rep4_seed22_29/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
mkdir -p "${E}"

run() {  # $1 = exhaustiveness, $2 = gpu
  conda run -n dif python "${S}" \
    --model DiffGui --input-sdf "${I}" --receptor "${ROOT}/data/8gjl.pdbqt" \
    --out-dir "${E}/exh$1" --unidock-bin /opt/unidock-1.1.3/bin/unidock \
    --gpu-id "$2" --batch-size 128 --exhaustiveness "$1" \
    >"${E}/exh$1.log" 2>&1
}

run 8 0 &
run 16 1 &
wait
echo "both finished"
