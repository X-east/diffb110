#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/analysis/diffsbdd_filtered_bc_epoch99_generate_8gpu_1000_260821}"
BASE_CHECKPOINT="$PROJECT_ROOT/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
OFFLINE_CHECKPOINT="$PROJECT_ROOT/collected_iql_benchmark_100k_260806/iql_worktree/analysis/diffsbdd_filtered_bc_high419_8gpu_100epoch_260821/ranks/rank_0000/checkpoints/epoch_0099_rank_0000.pt"
EXPORTED_CHECKPOINT="$RUN_DIR/checkpoints/diffsbdd_filtered_bc_epoch0099.ckpt"
MASTER_PORT="${MASTER_PORT:-29624}"

if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/shards" "$RUN_DIR/logs"

export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export PYTHONUNBUFFERED=1

conda run -n dif python "$PROJECT_ROOT/scripts/export_offline_actor_to_diffsbdd_checkpoint.py" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --offline-checkpoint "$OFFLINE_CHECKPOINT" \
    --output-checkpoint "$EXPORTED_CHECKPOINT" \
    --metadata-json "$RUN_DIR/checkpoints/export_metadata.json" \
    > "$RUN_DIR/logs/export.log" 2>&1

declare -a pids=()
for gpu_id in 0 1 2 3 4 5 6 7; do
    seed=$((gpu_id + 1))
    shard="$RUN_DIR/shards/gpu_$(printf '%02d' "$gpu_id").sdf"
    log="$RUN_DIR/logs/gpu_$(printf '%02d' "$gpu_id").log"
    CUDA_VISIBLE_DEVICES="$gpu_id" conda run -n dif python "$PROJECT_ROOT/DiffSBDD/generate_ligands.py" \
        "$EXPORTED_CHECKPOINT" \
        --pdbfile "$PROJECT_ROOT/data/8GJL.pdb" \
        --ref_ligand "C:1101" \
        --outfile "$shard" \
        --n_samples 125 \
        --batch_size 25 \
        --all_frags \
        --timesteps 1000 \
        --resamplings 10 \
        --jump_length 1 \
        --reconstruction_method openbabel \
        --seed "$seed" \
        > "$log" 2>&1 &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

for gpu_id in 0 1 2 3 4 5 6 7; do
    shard="$RUN_DIR/shards/gpu_$(printf '%02d' "$gpu_id").sdf"
    [ -s "$shard" ] || { echo "ERROR: missing or empty shard $shard" >&2; exit 1; }
done

cat "$RUN_DIR"/shards/gpu_*.sdf > "$RUN_DIR/generated_1000_attempts.sdf"
written_count=$(grep -c '^\$\$\$\$$' "$RUN_DIR/generated_1000_attempts.sdf" || true)
printf '{\n  "status": "completed",\n  "fixed_attempts": 1000,\n  "gpus": 8,\n  "attempts_per_gpu": 125,\n  "timesteps": 1000,\n  "seeds": [1, 2, 3, 4, 5, 6, 7, 8],\n  "written_sdf_records": %s\n}\n' "$written_count" \
    > "$RUN_DIR/completion.json"
