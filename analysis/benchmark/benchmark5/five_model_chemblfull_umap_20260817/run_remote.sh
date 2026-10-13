#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/fzmn1/DiffB110_five_model_chemblfull_20260817}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/fzmn1/anaconda3/envs/py38_g/bin/python}"
WORKERS="${WORKERS:-90}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
export NUMBA_CACHE_DIR="${ROOT_DIR}/.cache/numba"
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${ROOT_DIR}/outputs/analysis_v001"
exec taskset -c 0-89 "${PYTHON_BIN}" \
  "${ROOT_DIR}/pipeline/analyze_five_model_chemblfull.py" \
  --config "${ROOT_DIR}/pipeline/analysis_config.yaml" \
  --stage all
