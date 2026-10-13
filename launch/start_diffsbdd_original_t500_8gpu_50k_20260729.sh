#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
RUN_ID="${RUN_ID:-diffsbdd-original-t500-8gpu-50k_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_DIR:-${ROOT}/runs/diffsbdd_generation/${RUN_ID}}"
SNAPSHOT_DIR="${RUN_DIR}/source_snapshot"
CONFIG_DIR="${RUN_DIR}/config"

if [[ -e "${RUN_DIR}/launcher.pid" ]]; then
  echo "Refusing to overwrite existing launcher metadata: ${RUN_DIR}/launcher.pid" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/logs" "${SNAPSHOT_DIR}" "${CONFIG_DIR}"
cp "${ROOT}/launch/diffsbdd_original_t500_8gpu_50k_20260729.yaml" "${SNAPSHOT_DIR}/config.yaml"
cp "${ROOT}/launch/diffsbdd_original_t500_8gpu_50k_20260729.yaml" "${CONFIG_DIR}/resolved_config.yaml"
cp "${ROOT}/launch/run_diffsbdd_original_t500_8gpu_50k_20260729.sh" "${SNAPSHOT_DIR}/run_diffsbdd_original_t500_8gpu_50k_20260729.sh"
cp "${ROOT}/scripts/run_diffsbdd_50k_rank.py" "${SNAPSHOT_DIR}/run_diffsbdd_50k_rank.py"
cp "${ROOT}/scripts/summarize_diffsbdd_50k_run.py" "${SNAPSHOT_DIR}/summarize_diffsbdd_50k_run.py"
sha256sum "${SNAPSHOT_DIR}"/* > "${SNAPSHOT_DIR}/sha256sums.txt"
sha256sum \
  "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
  "${ROOT}/data/8GJL.pdb" \
  > "${CONFIG_DIR}/input_sha256s.txt"
git -C "${ROOT}" rev-parse HEAD > "${CONFIG_DIR}/git_commit.txt"
git -C "${ROOT}" status --short > "${CONFIG_DIR}/git_status_short.txt"
conda run -n dif python -c \
  "import sys, torch; print('python=' + sys.version.replace(chr(10), ' ')); print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); print('cuda_devices=' + str(torch.cuda.device_count()))" \
  > "${CONFIG_DIR}/environment.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "${CONFIG_DIR}/gpu_preflight.csv"

nohup bash "${SNAPSHOT_DIR}/run_diffsbdd_original_t500_8gpu_50k_20260729.sh" "${RUN_DIR}" \
  > "${RUN_DIR}/logs/launcher_stdout.log" 2>&1 &
launcher_pid=$!
echo "${launcher_pid}" > "${RUN_DIR}/launcher.pid"
printf '%s\n' "${RUN_DIR}"
printf 'launcher_pid=%s\n' "${launcher_pid}"
