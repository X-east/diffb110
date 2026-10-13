#!/usr/bin/env bash
set -euo pipefail

# DiffGui property conditioning strength test — 5 groups × 500 mols × 200 steps
# Sequential groups, each group runs independent per-GPU samplers.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFGUI_ROOT="/home/md666/data/disk6/DiffB110/reference project/代码/DiffGui"
CONFIG_DIR="$ROOT/configs"
OUTPUT_DIR="$ROOT/outputs"
LOG_DIR="$ROOT/logs"
CONDA_ENV="diffgui"
DRY_RUN="${DRY_RUN:-0}"
GPU_IDS="${GPU_IDS:-2 3 4 5 6 7}"
ONLY="${ONLY:-}"
START_AT="${START_AT:-}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Generate 5 base configs
conda run -n dif python "$ROOT/generate_configs.py"

for gpu_id in ${GPU_IDS}; do
    if [[ "$gpu_id" == "0" || "$gpu_id" == "1" ]]; then
        echo "[prop-test] ERROR: GPU_IDS must not include GPU 0 or GPU 1" >&2
        exit 1
    fi
done

echo "============================================================"
echo "[prop-test] 5 groups: zero / default / high_qed / simple_polar / affinity_push"
echo "[prop-test] GPUs: ${GPU_IDS}"
if [[ -n "$ONLY" ]]; then
    echo "[prop-test] ONLY: ${ONLY}"
fi
if [[ -n "$START_AT" ]]; then
    echo "[prop-test] START_AT: ${START_AT}"
fi
echo "[prop-test] each: 200 steps, ${GPU_IDS} × 84 = ~504 mols before filtering"
echo "============================================================"

mapfile -t base_yamls < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yml' | sort)
started=0
if [[ -z "$START_AT" ]]; then
    started=1
fi

for base_yaml in "${base_yamls[@]}"; do
    base_name="$(basename "$base_yaml" .yml)"
    if [[ -n "$ONLY" && "$base_name" != "$ONLY" ]]; then
        continue
    fi
    if [[ "$started" == "0" ]]; then
        if [[ "$base_name" == "$START_AT" ]]; then
            started=1
        else
            continue
        fi
    fi
    group_dir="$OUTPUT_DIR/$base_name"
    group_log="$LOG_DIR/${base_name}.log"
    mkdir -p "$group_dir"

    echo "============================================================"
    echo "[prop-test] === $base_name ==="
    echo "[prop-test] start: $(date '+%F %T')"
    echo "============================================================"
    : > "$group_log"

    pids=()
    for gpu_id in ${GPU_IDS}; do
        seed=$((20260616 + gpu_id * 100))
        gpu_dir="$group_dir/gpu_${gpu_id}"
        mkdir -p "$gpu_dir"
        # Each GPU config — same base, different seed
        gpu_config="$gpu_dir/config.yml"
        sed "s/seed: 20260616/seed: $seed/" "$base_yaml" > "$gpu_config"

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[prop-test] DRY_RUN: GPU=$gpu_id seed=$seed config=$gpu_config"
            continue
        fi

        (
            cd "$DIFFGUI_ROOT"
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            conda run -n "$CONDA_ENV" python scripts/sample.py \
                --config "$gpu_config" \
                --device "cuda:0" \
                --outdir "$gpu_dir" \
                >> "$group_log" 2>&1
        ) &
        pids+=($!)
    done

    if [[ "$DRY_RUN" == "1" ]]; then continue; fi

    group_status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[prop-test] child pid=$pid failed for $base_name" | tee -a "$group_log"
            group_status=1
        fi
    done
    if [[ "$group_status" != "0" ]]; then
        echo "[prop-test] ERROR: one or more GPU workers failed for $base_name" | tee -a "$group_log"
        exit "$group_status"
    fi

    # Aggregate SDFs (in timestamped config dir, under *_SDF/)
    agg_sdf="$group_dir/aggregated.sdf"
    manifest_csv="$group_dir/aggregated_manifest.csv"
    conda run -n dif python "$ROOT/aggregate_sdfs.py" \
        --group-dir "$group_dir" \
        --gpu-ids ${GPU_IDS} \
        --out "$agg_sdf" \
        --manifest "$manifest_csv"
    n_total=$(grep -c '\$\$\$\$' "$agg_sdf" 2>/dev/null || echo 0)
    echo "[prop-test] $base_name: $n_total molecules → $agg_sdf"
    echo "[prop-test] done  $base_name  end=$(date '+%F %T')"
done

echo "============================================================"
echo "[prop-test] ALL DONE at $(date '+%F %T')"
echo "[prop-test] outputs: $OUTPUT_DIR"
echo "[prop-test] run analysis: conda run -n dif python $ROOT/analyze_results.py"
echo "============================================================"
