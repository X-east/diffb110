#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="$ROOT/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/generation_comparison"
LOG="$ROOT/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/docking_matrix.log"
STATUS="$ROOT/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/docking_matrix_status.json"

mkdir -p "$OUT_ROOT"
printf '{"status":"running","started_at":"%s","methods":["original","x_only","v_only","native_joint"],"attempts_per_method":1000,"gpu_ids":{"original":0,"x_only":1,"v_only":2,"native_joint":3}}\n' "$(date -Is)" > "$STATUS"

run_one() {
    local method="$1"
    local gpu="$2"
    local input="$OUT_ROOT/$method/molecules.sdf"
    local output="$OUT_ROOT/$method/docking"
    local log="$OUT_ROOT/$method/docking.log"
    echo "$(date -Is) START method=$method gpu=$gpu input=$input output=$output" >> "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n dif python "$ROOT/scripts/docking/dock_sdf_unidock.py" \
        --model "E2PO_${method}" \
        --input-sdf "$input" \
        --receptor "$ROOT/data/8gjl.pdbqt" \
        --out-dir "$output" \
        --center-x 207.293 --center-y 212.430 --center-z 172.771 \
        --size-x 20 --size-y 20 --size-z 20 \
        --gpu-id "$gpu" --batch-size 128 --search-mode fast --scoring vina --num-modes 1 --seed "$((gpu + 1))" \
        > "$log" 2>&1
    echo "$(date -Is) DONE method=$method gpu=$gpu" >> "$LOG"
}

run_one original 0 & p_original=$!
run_one x_only 1 & p_x_only=$!
run_one v_only 2 & p_v_only=$!
run_one native_joint 3 & p_native_joint=$!

failed=0
wait "$p_original" || failed=1
wait "$p_x_only" || failed=1
wait "$p_v_only" || failed=1
wait "$p_native_joint" || failed=1

if [[ "$failed" -eq 0 ]]; then
    printf '{"status":"completed","completed_at":"%s","attempts_per_method":1000}\n' "$(date -Is)" > "$STATUS"
else
    printf '{"status":"failed","completed_at":"%s","attempts_per_method":1000}\n' "$(date -Is)" > "$STATUS"
    exit 1
fi
