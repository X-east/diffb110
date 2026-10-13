#!/usr/bin/env bash
# ChEMBL docking smoke: 8 shards, one UniDock --ligand_index call per shard.
#
# Each shard runs as its own process pinned to one GPU. Input is the per-molecule
# SDF set produced by the pilot (also used by the old 128-batch run, so scores can
# be compared A/B at identical seed).
#
# Run detached:
#   setsid nohup bash analysis/chembl_docking_reference_260928/run_smoke_8gpu.sh \
#     > analysis/chembl_docking_reference_260928/smoke_wrapper.log 2>&1 < /dev/null &
set -u

cd /home/md666/data/disk6/DiffB110 || exit 1
PY=/home/md666/miniforge3/envs/dif/bin/python
D=analysis/chembl_docking_reference_260928
LIG="$D/pilot_dock/split_ligands"
OUT="$D/smoke_dock"
NUM_SHARDS=8

mkdir -p "$OUT"
echo "START $(date)" > "$OUT/run_status.txt"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" "$D/dock_shard_index.py" \
    --ligand-dir "$LIG" \
    --shard-id "$i" --num-shards "$NUM_SHARDS" \
    --receptor data/8gjl.pdbqt \
    --center-x 207.293 --center-y 212.430 --center-z 172.771 \
    --size-x 20 --size-y 20 --size-z 20 \
    --scoring vina --search-mode fast --num-modes 1 --seed 42 \
    --max-gpu-memory 0 \
    --out-dir "$OUT/shard_$i" > "$OUT/shard_$i.launch.log" 2>&1 &
done

wait
echo "ALL_DONE $(date)" >> "$OUT/run_status.txt"
