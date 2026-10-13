#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/disk6/DiffB110"
RUN_DIR="${ROOT}/analysis/single_side_fragment_growth_20260718"

env -C "${ROOT}/reference project/代码/modern_sbdd_20260531/PMDM" \
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 16-19 conda run -n pmdm_modern_ref python \
  "${RUN_DIR}/run_pmdm_script.py" \
  "${ROOT}/reference project/代码/modern_sbdd_20260531/PMDM" sample_frag.py \
  --ckpt "${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/PMDM/500.pt" \
  --pdb_path "${RUN_DIR}/pmdm/input/4dxd_honokiol_pocket10A.pdb" \
  --mol_file "${RUN_DIR}/pmdm/input/fixed_core_no_oh.sdf" \
  --keep_index 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \
  --num_samples 20 \
  --batch_size 1 \
  --n_steps 100 \
  --sampling_type generalized \
  --seed 4 \
  -build_method build \
  > "${RUN_DIR}/pmdm/generation_build.log" 2>&1
