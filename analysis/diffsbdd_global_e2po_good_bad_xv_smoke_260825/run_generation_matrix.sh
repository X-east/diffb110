#!/usr/bin/env bash
set -euo pipefail

# Fixed-attempt generation matrix for the original and three completed DPO runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${ROOT}/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/generation_comparison"
BASE_CHECKPOINT="${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
POCKET="${ROOT}/data/8GJL.pdb"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"
ATTEMPTS="${ATTEMPTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-25}"
TIMESTEPS="${TIMESTEPS:-500}"
SEED="${SEED:-1}"
RESAMPLINGS="${RESAMPLINGS:-10}"
JUMP_LENGTH="${JUMP_LENGTH:-1}"
RECONSTRUCTION_METHOD="${RECONSTRUCTION_METHOD:-openbabel}"

mkdir -p "${RUN_DIR}"

write_status() {
  local path="$1"
  local status="$2"
  local extra="${3:-}"
  printf '{\n  "status": "%s",\n  "updated_at": "%s"%s\n}\n' \
    "${status}" "$(date -Is)" "${extra}" > "${path}"
}

run_one() {
  local name="$1"
  local gpu="$2"
  local dpo_checkpoint="$3"
  local out_dir="${RUN_DIR}/${name}"
  local log_path="${out_dir}/generation.log"
  local status_path="${out_dir}/status.json"
  local generation_checkpoint="${out_dir}/generation.ckpt"
  local sdf_path="${out_dir}/molecules.sdf"

  mkdir -p "${out_dir}"
  write_status "${status_path}" "running" ",\n  \"gpu\": ${gpu},\n  \"fixed_attempts\": ${ATTEMPTS},\n  \"timesteps\": ${TIMESTEPS}"
  {
    echo "$(date -Is) INFO name=${name} gpu=${gpu} attempts=${ATTEMPTS} timesteps=${TIMESTEPS} seed=${SEED}"
    echo "$(date -Is) INFO output=${out_dir}"
    if [[ "${dpo_checkpoint}" != "-" ]]; then
      echo "$(date -Is) INFO exporting dpo_checkpoint=${dpo_checkpoint}"
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}:${ROOT}/DiffSBDD" \
        "${PYTHON_BIN}" "${ROOT}/scripts/export_diffsbdd_global_dpo_checkpoint.py" \
        --dpo-checkpoint "${dpo_checkpoint}" \
        --base-checkpoint "${BASE_CHECKPOINT}" \
        --output "${generation_checkpoint}"
      checkpoint_for_generation="${generation_checkpoint}"
    else
      checkpoint_for_generation="${BASE_CHECKPOINT}"
    fi
    echo "$(date -Is) INFO generation_checkpoint=${checkpoint_for_generation}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}:${ROOT}/DiffSBDD" \
      BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0" \
      PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${ROOT}/DiffSBDD/generate_ligands.py" \
      "${checkpoint_for_generation}" \
      --pdbfile "${POCKET}" \
      --ref_ligand "C:1101" \
      --outfile "${sdf_path}" \
      --n_samples "${ATTEMPTS}" \
      --batch_size "${BATCH_SIZE}" \
      --timesteps "${TIMESTEPS}" \
      --seed "${SEED}" \
      --resamplings "${RESAMPLINGS}" \
      --jump_length "${JUMP_LENGTH}" \
      --reconstruction_method "${RECONSTRUCTION_METHOD}" \
      --all_frags
    written_records="$(awk 'BEGIN { n=0 } /^\$\$\$\$$/ { n++ } END { print n }' "${sdf_path}")"
    printf '{\n  "status": "completed",\n  "completed_at": "%s",\n  "gpu": %s,\n  "fixed_attempts": %s,\n  "written_sdf_records": %s,\n  "seed": %s,\n  "timesteps": %s,\n  "batch_size": %s,\n  "checkpoint": "%s",\n  "molecules_sdf": "%s"\n}\n' \
      "$(date -Is)" "${gpu}" "${ATTEMPTS}" "${written_records}" "${SEED}" "${TIMESTEPS}" "${BATCH_SIZE}" \
      "${checkpoint_for_generation}" "${sdf_path}" > "${status_path}"
    echo "$(date -Is) INFO completed written_sdf_records=${written_records}"
  } > "${log_path}" 2>&1 || {
    write_status "${status_path}" "failed" ",\n  \"gpu\": ${gpu},\n  \"fixed_attempts\": ${ATTEMPTS}"
    return 1
  }
}

run_one original 0 - &
pid_original=$!
run_one x_only 1 "${ROOT}/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/x_only/checkpoints/diffsbdd-global-dpo-epoch=0001.pt" &
pid_x_only=$!
run_one v_only 2 "${ROOT}/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/v_only/checkpoints/diffsbdd-global-dpo-epoch=0001.pt" &
pid_v_only=$!
run_one native_joint 3 "${ROOT}/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/native_joint/checkpoints/diffsbdd-global-dpo-epoch=0001.pt" &
pid_native_joint=$!

printf '{\n  "status": "running",\n  "started_at": "%s",\n  "fixed_attempts_per_method": %s,\n  "timesteps": %s,\n  "seed": %s,\n  "gpus": {"original": 0, "x_only": 1, "v_only": 2, "native_joint": 3},\n  "pids": {"original": %s, "x_only": %s, "v_only": %s, "native_joint": %s}\n}\n' \
  "$(date -Is)" "${ATTEMPTS}" "${TIMESTEPS}" "${SEED}" \
  "${pid_original}" "${pid_x_only}" "${pid_v_only}" "${pid_native_joint}" \
  > "${RUN_DIR}/matrix_status.json"
printf '%s\n' "$pid_original" > "${RUN_DIR}/original.pid"
printf '%s\n' "$pid_x_only" > "${RUN_DIR}/x_only.pid"
printf '%s\n' "$pid_v_only" > "${RUN_DIR}/v_only.pid"
printf '%s\n' "$pid_native_joint" > "${RUN_DIR}/native_joint.pid"

failed=0
wait "${pid_original}" || failed=1
wait "${pid_x_only}" || failed=1
wait "${pid_v_only}" || failed=1
wait "${pid_native_joint}" || failed=1

if [[ "${failed}" -eq 0 ]]; then
  printf '{\n  "status": "completed",\n  "completed_at": "%s",\n  "fixed_attempts_per_method": %s,\n  "timesteps": %s,\n  "seed": %s\n}\n' \
    "$(date -Is)" "${ATTEMPTS}" "${TIMESTEPS}" "${SEED}" > "${RUN_DIR}/matrix_status.json"
else
  printf '{\n  "status": "failed",\n  "updated_at": "%s",\n  "fixed_attempts_per_method": %s,\n  "timesteps": %s,\n  "seed": %s\n}\n' \
    "$(date -Is)" "${ATTEMPTS}" "${TIMESTEPS}" "${SEED}" > "${RUN_DIR}/matrix_status.json"
  exit 1
fi
