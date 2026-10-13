#!/usr/bin/env bash
# DiffGui generation at T=200, otherwise identical to the T=500 rep4 batch
# (8 shards x 125 attempts = 1000, seeds 22-29, same pocket, same configs).
# Purpose: show that 200 diffusion steps is statistically equivalent to 500.
set -uo pipefail
source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT=/home/md666/data/disk6/DiffB110
BASE_EXP="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps"
SRC_REP="${ROOT}/analysis/benchmark/benchmark5/five_model_repeats_500steps_260826/rep4_seed22_29"
EXP="${ROOT}/analysis/benchmark/benchmark5/diffgui_200steps_validity_260918"
PROJECT="${ROOT}/reference project/代码/DiffGui"

mkdir -p "${EXP}/configs/diffgui" "${EXP}/logs" "${EXP}/mplconfig"
LOG="${EXP}/logs/generation.log"
: >"${LOG}"

SEEDS=(22 23 24 25 26 27 28 29)

for i in 0 1 2 3 4 5 6 7; do
  shard=$((i + 1))
  seed=${SEEDS[$i]}
  cfg="${EXP}/configs/diffgui/diffgui_c1101_gpu${shard}.yml"
  sed -E "s/num_timesteps: 500/num_timesteps: 200/; s/seed: [0-9]+/seed: ${seed}/" \
    "${SRC_REP}/configs/diffgui/diffgui_c1101_gpu${shard}.yml" >"${cfg}"
done
echo "[$(date -Is)] wrote 8 configs (num_timesteps=200, seeds 22-29)" | tee -a "${LOG}"

run_shard() {
  local gpu="$1" shard="$2"
  local out="${EXP}/outputs/diffgui/gpu_${shard}"
  mkdir -p "${out}"
  (
    cd "${PROJECT}" || exit 1
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export DIFFGUI_NUM_TIMESTEPS=200
    export PYTHONPATH="${BASE_EXP}/runtime_patch/diffgui:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
    export MPLCONFIGDIR="${EXP}/mplconfig/gpu_${shard}"
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    conda run -n diffgui python scripts/sample.py \
      --config "${EXP}/configs/diffgui/diffgui_c1101_gpu${shard}.yml" \
      --outdir "${out}" --device cuda:0 --batch_size 2 --fixed_attempts \
      --seed "${SEEDS[$((shard - 1))]}"
  ) >"${EXP}/logs/shard_${shard}.log" 2>&1
  echo "[$(date -Is)] shard ${shard} (gpu ${gpu}) exit=$?" >>"${LOG}"
}

# only GPUs 0-4 are free; distribute the 8 shards over them
for i in 0 1 2 3 4 5 6 7; do
  run_shard "$((i % 5))" "$((i + 1))" &
done
wait
echo "[$(date -Is)] all shards done" | tee -a "${LOG}"
