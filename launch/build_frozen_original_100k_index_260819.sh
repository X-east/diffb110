#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/disk6/DiffB110"
WORKTREE="${ROOT}/collected_iql_benchmark_100k_260806/iql_worktree"
OUT_DIR="${ROOT}/analysis/offline_iql_frozen_original_100k_260819/index"

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${WORKTREE}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${OUT_DIR}/runtime_cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

exec conda run -n dif python "${WORKTREE}/scripts/offline_rl/build_episode_index.py" \
  --run-dir "${ROOT}/analysis/offline_iql_frozen_original_100k_260819/run_view" \
  --output-dir "${OUT_DIR}" \
  --max-shards-per-rank 1250 \
  --write-unscored-split \
  --split-version "frozen_original_100k_unscored_v1" \
  --split-salt "frozen-original-100k-v1" \
  --progress-every-shards 250
