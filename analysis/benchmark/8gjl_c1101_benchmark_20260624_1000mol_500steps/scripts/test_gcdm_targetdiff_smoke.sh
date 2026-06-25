#!/usr/bin/env bash
# Smoke-test only GCDM-SBDD and TargetDiff for the 8GJL/C1101 benchmark.
# This script does not rerun DiffGui, DiffSBDD, PMDM, or any evaluation step.
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="/home/md666/data/disk6/DiffB110"
EXP="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps"
INPUTS="${EXP}/inputs"

GPU_ID="${GPU_ID:-1}"
N_SAMPLES="${N_SAMPLES:-8}"
TIMESTEPS="${TIMESTEPS:-500}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${EXP}/work/gcdm_targetdiff_smoke_${RUN_TAG}}"
TARGETDIFF_BATCH_SIZE="${TARGETDIFF_BATCH_SIZE:-5}"

GCDM_PROJECT="${ROOT}/reference project/modern_sbdd_20260531/GCDM-SBDD"
GCDM_CKPT="${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/GCDM-SBDD/extracted/checkpoints/crossdocked_ca_joint_egnn.ckpt"
GCDM_OUT="${OUT_ROOT}/gcdm"
GCDM_LOG="${OUT_ROOT}/logs/gcdm.log"

TARGETDIFF_PROJECT="/home/md666/data/disk6/XDC/targetdiff"
TARGETDIFF_CKPT="/home/md666/data/disk6/XDC/targetdiff/pretrained_models/pretrained_diffusion.pt"
TARGETDIFF_CONFIG="${OUT_ROOT}/targetdiff_smoke.yml"
TARGETDIFF_OUT="${OUT_ROOT}/targetdiff"
TARGETDIFF_LOG="${OUT_ROOT}/logs/targetdiff.log"

mkdir -p "${OUT_ROOT}/logs" "${GCDM_OUT}" "${TARGETDIFF_OUT}"

echo "[smoke] started at $(date -Is)"
echo "[smoke] gpu=${GPU_ID} n_samples=${N_SAMPLES} timesteps=${TIMESTEPS} targetdiff_batch_size=${TARGETDIFF_BATCH_SIZE}"
echo "[smoke] out_root=${OUT_ROOT}"

echo "[smoke] running GCDM-SBDD..."
(
  cd "${GCDM_PROJECT}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  conda run -n gcdm_sbdd_modern_ref python generate_ligands.py \
    "${GCDM_CKPT}" \
    --pdbfile "${ROOT}/data/8GJL.pdb" \
    --ref_ligand "C:1101" \
    --outdir "${GCDM_OUT}" \
    --n_samples "${N_SAMPLES}" \
    --timesteps "${TIMESTEPS}" \
    --seed 101
) >"${GCDM_LOG}" 2>&1

GCDM_SDF="${GCDM_OUT}/8GJL_mol.sdf"
if [[ ! -s "${GCDM_SDF}" ]]; then
  echo "[smoke] GCDM-SBDD failed: expected non-empty ${GCDM_SDF}" >&2
  echo "[smoke] see ${GCDM_LOG}" >&2
  exit 1
fi
echo "[smoke] GCDM-SBDD ok: ${GCDM_SDF}"

cat >"${TARGETDIFF_CONFIG}" <<EOF
model:
  checkpoint: ${TARGETDIFF_CKPT}
input:
  protein_path: ${INPUTS}/8GJL_C1101_protein_only.pdb
  ligand_path: ${INPUTS}/8GJL_C1101_ligand.sdf
  pocket_radius: 10.0
runtime:
  device: cuda:0
  batch_size: ${TARGETDIFF_BATCH_SIZE}
  result_path: ${TARGETDIFF_OUT}
sample:
  seed: 20260625
  num_samples: ${N_SAMPLES}
  num_steps: ${TIMESTEPS}
  pos_only: false
  center_pos_mode: protein
  sample_num_atoms: prior
EOF

echo "[smoke] running TargetDiff..."
(
  cd "${TARGETDIFF_PROJECT}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  conda run -n targetdiff python scripts/sample_from_protein_ligand.py \
    "${TARGETDIFF_CONFIG}"
) >"${TARGETDIFF_LOG}" 2>&1

TARGETDIFF_SDF="${TARGETDIFF_OUT}/generated_molecules.sdf"
if [[ ! -s "${TARGETDIFF_SDF}" ]]; then
  echo "[smoke] TargetDiff failed: expected non-empty ${TARGETDIFF_SDF}" >&2
  echo "[smoke] see ${TARGETDIFF_LOG}" >&2
  exit 1
fi
echo "[smoke] TargetDiff ok: ${TARGETDIFF_SDF}"

echo "[smoke] completed at $(date -Is)"
echo "[smoke] logs:"
echo "  GCDM-SBDD: ${GCDM_LOG}"
echo "  TargetDiff: ${TARGETDIFF_LOG}"
