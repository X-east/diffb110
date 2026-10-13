#!/usr/bin/env bash
# Export sanitize-valid subsets for the 8GJL/C1101 benchmark and run property plots.
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="/data/disk6/DiffB110"
EXP="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps"
SCRIPTS="${EXP}/scripts"

RAW_EVAL_DIR="${EXP}/evaluation/raw_all_models"
VALID_DIR="${EXP}/filtered_sanitize_valid"
VALID_EVAL_DIR="${EXP}/evaluation/sanitize_valid_only"

MODELS=(
  "diffgui=${EXP}/aggregated/diffgui.sdf"
  "diffsbdd=${EXP}/aggregated/diffsbdd.sdf"
  "pmdm=${EXP}/aggregated/pmdm.sdf"
  "gcdm=${EXP}/aggregated/gcdm.sdf"
  "targetdiff=${EXP}/aggregated/targetdiff.sdf"
  "graphbp=${EXP}/aggregated/graphbp.sdf"
  "pocket2mol=${EXP}/aggregated/pocket2mol.sdf"
)

conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
  --out-dir "${RAW_EVAL_DIR}" \
  --sdf "${MODELS[0]}" \
  --sdf "${MODELS[1]}" \
  --sdf "${MODELS[2]}" \
  --sdf "${MODELS[3]}" \
  --sdf "${MODELS[4]}" \
  --sdf "${MODELS[5]}" \
  --sdf "${MODELS[6]}"

conda run -n dif python "${SCRIPTS}/export_sanitize_valid_sdfs.py" \
  --out-dir "${VALID_DIR}" \
  --sdf "${MODELS[0]}" \
  --sdf "${MODELS[1]}" \
  --sdf "${MODELS[2]}" \
  --sdf "${MODELS[3]}" \
  --sdf "${MODELS[4]}" \
  --sdf "${MODELS[5]}" \
  --sdf "${MODELS[6]}"

conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
  --out-dir "${VALID_EVAL_DIR}" \
  --sdf "diffgui=${VALID_DIR}/diffgui.sanitize_valid.sdf" \
  --sdf "diffsbdd=${VALID_DIR}/diffsbdd.sanitize_valid.sdf" \
  --sdf "pmdm=${VALID_DIR}/pmdm.sanitize_valid.sdf" \
  --sdf "gcdm=${VALID_DIR}/gcdm.sanitize_valid.sdf" \
  --sdf "targetdiff=${VALID_DIR}/targetdiff.sanitize_valid.sdf" \
  --sdf "graphbp=${VALID_DIR}/graphbp.sanitize_valid.sdf" \
  --sdf "pocket2mol=${VALID_DIR}/pocket2mol.sanitize_valid.sdf"
