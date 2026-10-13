#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT_ROOT="$SCRIPT_DIR/docking_260826"
LOG="$OUT_ROOT/docking_launcher.log"
STATUS="$OUT_ROOT/docking_status.json"

mkdir -p "$OUT_ROOT"
printf '{"status":"running","started_at":"%s","methods":["original","ecfp4_e2po"],"attempts_per_method":1000,"receptor":"%s","box":{"center":[207.293,212.430,172.771],"size":[20,20,20]}}\n' "$(date -Is)" "$ROOT/data/8gjl.pdbqt" > "$STATUS"

run_one() {
    local method="$1"
    local gpu="$2"
    local seed="$3"
    local input_sdf="$4"
    local output="$OUT_ROOT/$method"
    local log="$output/docking.log"
    mkdir -p "$output"
    echo "$(date -Is) START method=$method gpu=$gpu seed=$seed input=$input_sdf output=$output" >> "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n dif python "$ROOT/scripts/docking/dock_sdf_unidock.py" \
        --model "ECFP4_${method}" \
        --input-sdf "$input_sdf" \
        --receptor "$ROOT/data/8gjl.pdbqt" \
        --out-dir "$output" \
        --center-x 207.293 --center-y 212.430 --center-z 172.771 \
        --size-x 20 --size-y 20 --size-z 20 \
        --gpu-id "$gpu" --batch-size 128 --search-mode fast --scoring vina --num-modes 1 --seed "$seed" \
        > "$log" 2>&1
    echo "$(date -Is) DONE method=$method gpu=$gpu seed=$seed" >> "$LOG"
}

run_one original 0 1 "$ROOT/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/generation_comparison/original/molecules.sdf" &
p_original=$!
run_one ecfp4_e2po 1 2 "$SCRIPT_DIR/generation/molecules.sdf" &
p_ecfp4_e2po=$!

failed=0
wait "$p_original" || failed=1
wait "$p_ecfp4_e2po" || failed=1

if [[ "$failed" -eq 0 ]]; then
    printf '{"status":"completed","completed_at":"%s","attempts_per_method":1000}\n' "$(date -Is)" > "$STATUS"
else
    printf '{"status":"failed","completed_at":"%s","attempts_per_method":1000}\n' "$(date -Is)" > "$STATUS"
    exit 1
fi
