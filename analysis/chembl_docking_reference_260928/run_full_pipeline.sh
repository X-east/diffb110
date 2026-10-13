#!/usr/bin/env bash
# ChEMBL -> RE-CmeB docking reference: pipelined full run.
#
# Stage 1 (CPU): stream 3D ETKDG+MMFF shards of SHARD_SIZE molecules each.
# Stage 2 (GPU): NGPU workers claim completed shards and dock them with a single
#                UniDock --ligand_index call each (auto-batched by GPU memory).
#
# Runnable detached:
#   setsid nohup bash run_full_pipeline.sh > <out>/wrapper.log 2>&1 < /dev/null &
#
# Env overrides (used for the smoke):
#   LIMIT, SHARD_SIZE, NGPU, MMFF_ITERS, NUM_WORKERS, OUT
set -u

cd /home/md666/data/disk6/DiffB110 || exit 1
PY=/home/md666/miniforge3/envs/dif/bin/python
D=analysis/chembl_docking_reference_260928

LIMIT=${LIMIT:-0}
SHARD_SIZE=${SHARD_SIZE:-20000}
NGPU=${NGPU:-8}
MMFF_ITERS=${MMFF_ITERS:-200}
NUM_WORKERS=${NUM_WORKERS:-16}
REDO_SHARDS=${REDO_SHARDS:-}
REDO_TAG=${REDO_TAG:-$(date +%s)}
GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
OUT=${OUT:-$D/full}

mkdir -p "$OUT"
echo "START $(date)  LIMIT=$LIMIT SHARD_SIZE=$SHARD_SIZE NGPU=$NGPU MMFF_ITERS=$MMFF_ITERS" \
  > "$OUT/run_status.txt"

# ---- Stage 1: 3D generation ------------------------------------------------
"$PY" "$D/gen_3d_shards.py" \
  --input data/chembl_smallmol_reference/reference_smallmol.csv.gz \
  --out-dir "$OUT" \
  --shard-size "$SHARD_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-heavy-atoms 60 \
  --mmff-max-iters "$MMFF_ITERS" \
  --limit "$LIMIT" \
  ${REDO_SHARDS:+--redo-shards "$REDO_SHARDS"} \
  > "$OUT/gen.log" 2>&1 &
GEN_PID=$!
echo "GEN_PID=$GEN_PID" >> "$OUT/run_status.txt"

# ---- Stage 2: docking workers ---------------------------------------------
DOCK_PIDS=()
for gpu in $GPU_LIST; do
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$D/dock_worker.py" \
    --out-root "$OUT" --gpu-id "$gpu" \
    --redo-tag "$REDO_TAG" \
    > "$OUT/dock_gpu$gpu.log" 2>&1 &
  DOCK_PIDS+=($!)
done
echo "DOCK_PIDS=${DOCK_PIDS[*]}  GPU_LIST=$GPU_LIST" >> "$OUT/run_status.txt"

fail=0
wait "$GEN_PID" || fail=1
for pid in "${DOCK_PIDS[@]}"; do
  wait "$pid" || fail=1
done
echo "ALL_DONE $(date) fail=$fail" >> "$OUT/run_status.txt"
