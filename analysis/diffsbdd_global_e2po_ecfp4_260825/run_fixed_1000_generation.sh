#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/disk6/DiffB110"
OUT="${ROOT}/analysis/diffsbdd_global_e2po_ecfp4_260825/generation_native_joint"
GEN_DIR="${OUT}/generation"
STATUS="${OUT}/status.json"
DPO_CHECKPOINT="${ROOT}/analysis/diffsbdd_global_e2po_ecfp4_260825/native_joint_smoke/checkpoints/diffsbdd-global-dpo-epoch=0001.pt"
BASE_CHECKPOINT="${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
GENERATION_CHECKPOINT="${GEN_DIR}/global_dpo_generation.ckpt"
SDF_PATH="${GEN_DIR}/molecules.sdf"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"

mkdir -p "${GEN_DIR}"

write_status() {
  local state="$1"
  local extra="${2:-}"
  printf '{\n  "status": "%s",\n  "updated_at": "%s"%s\n}\n' \
    "${state}" "$(date -Is)" "${extra}" > "${STATUS}"
}

on_error() {
  local exit_code="$?"
  write_status "failed" ",\n  \"exit_code\": ${exit_code}"
  exit "${exit_code}"
}
trap on_error ERR

write_status "running" ",\n  \"fixed_attempts\": 1000,\n  \"seed\": 1,\n  \"timesteps\": 500,\n  \"gpu\": 0"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${ROOT}:${ROOT}/DiffSBDD" \
  "${PYTHON_BIN}" "${ROOT}/scripts/export_diffsbdd_global_dpo_checkpoint.py" \
  --dpo-checkpoint "${DPO_CHECKPOINT}" \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --output "${GENERATION_CHECKPOINT}" \
  > "${GEN_DIR}/export.log" 2>&1

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${ROOT}:${ROOT}/DiffSBDD" \
  BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0" \
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${ROOT}/DiffSBDD/generate_ligands.py" \
  "${GENERATION_CHECKPOINT}" \
  --pdbfile "${ROOT}/data/8GJL.pdb" \
  --ref_ligand "C:1101" \
  --outfile "${SDF_PATH}" \
  --n_samples 1000 \
  --batch_size 25 \
  --timesteps 500 \
  --seed 1 \
  --resamplings 10 \
  --jump_length 1 \
  --reconstruction_method openbabel \
  --all_frags \
  > "${GEN_DIR}/generation.log" 2>&1

WRITTEN_RECORDS="$(awk 'BEGIN { n=0 } /^\$\$\$\$$/ { n++ } END { print n }' "${SDF_PATH}")"
printf '{\n  "status": "completed",\n  "completed_at": "%s",\n  "fixed_attempts": 1000,\n  "written_sdf_records": %s,\n  "seed": 1,\n  "timesteps": 500,\n  "batch_size": 25,\n  "gpu": 0,\n  "generation_checkpoint": "%s",\n  "molecules_sdf": "%s"\n}\n' \
  "$(date -Is)" "${WRITTEN_RECORDS}" "${GENERATION_CHECKPOINT}" "${SDF_PATH}" > "${STATUS}"
