#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${DIFFB110_PROJECT_ROOT:-/data/disk6/DiffB110}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901}"
DPO_CHECKPOINT="${DPO_CHECKPOINT:-${PROJECT_ROOT}/analysis/diffsbdd_global_e2po_structure_matched_x_only_test_260901/checkpoints/diffsbdd-global-dpo-epoch=0001.pt}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT_ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt}"
GEN_CHECKPOINT="${RUN_DIR}/global_dpo_generation.ckpt"
SDF_PATH="${RUN_DIR}/molecules.sdf"
GPU_ID="${GPU_ID:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"
ATTEMPTS="${ATTEMPTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-25}"

mkdir -p "${RUN_DIR}"
if [[ ! -f "${DPO_CHECKPOINT}" || ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "ERROR missing checkpoint input" >&2
  exit 2
fi
if [[ -e "${SDF_PATH}" || -e "${GEN_CHECKPOINT}" ]]; then
  echo "ERROR generation output already exists: ${RUN_DIR}" >&2
  exit 2
fi

echo "$(date -Is) INFO generation_start launcher_pid=$$ physical_gpu=${GPU_ID} attempts=${ATTEMPTS} batch_size=${BATCH_SIZE}"
echo "$(date -Is) INFO dpo_checkpoint=${DPO_CHECKPOINT}"
echo "$(date -Is) INFO output_sdf=${SDF_PATH}"

conda run --no-capture-output -n dif python "${PROJECT_ROOT}/scripts/export_diffsbdd_global_dpo_checkpoint.py" \
  --dpo-checkpoint "${DPO_CHECKPOINT}" \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --output "${GEN_CHECKPOINT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/DiffSBDD${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/DiffSBDD/generate_ligands.py" "${GEN_CHECKPOINT}" \
  --pdbfile "${PROJECT_ROOT}/data/8GJL.pdb" \
  --ref_ligand "C:1101" \
  --outfile "${SDF_PATH}" \
  --n_samples "${ATTEMPTS}" \
  --batch_size "${BATCH_SIZE}" \
  --timesteps 500 \
  --seed 1 \
  --resamplings 10 \
  --jump_length 1 \
  --reconstruction_method openbabel \
  --all_frags

echo "$(date -Is) INFO generation_completed attempts=${ATTEMPTS} output=${SDF_PATH}"
