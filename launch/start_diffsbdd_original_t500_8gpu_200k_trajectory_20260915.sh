#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"
CONFIG_SOURCE="${ROOT}/launch/diffsbdd_original_t500_8gpu_200k_trajectory_20260915.yaml"
RUN_SOURCE="${ROOT}/launch/run_diffsbdd_original_t500_8gpu_200k_trajectory_20260915.sh"
ENTRY_SOURCE="${ROOT}/ddpo/main_diffsbdd_collect_ddp.py"
VALIDATOR_SOURCE="${ROOT}/scripts/validate_diffsbdd_trajectory_run.py"
NEW_RUN_DIR="${ROOT}/runs/diffsbdd_collect/diffsbdd-original-t500-8gpu-200k-trajectory_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESUME_RUN_DIR:-${RUN_DIR:-${NEW_RUN_DIR}}}"

for command_name in cp find flock git nvidia-smi setsid sha256sum sort xargs; do
  command -v "${command_name}" >/dev/null || {
    echo "ERROR required command not found: ${command_name}" >&2
    exit 1
  }
done
for required_file in \
  "${PYTHON_BIN}" \
  "${CONFIG_SOURCE}" \
  "${RUN_SOURCE}" \
  "${ENTRY_SOURCE}" \
  "${VALIDATOR_SOURCE}" \
  "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
  "${ROOT}/data/8GJL.pdb"; do
  if [[ ! -e "${required_file}" ]]; then
    echo "ERROR required input not found: ${required_file}" >&2
    exit 1
  fi
done
bash -n "${RUN_SOURCE}"
bash -n "$0"
"${PYTHON_BIN}" -c "import yaml; c=yaml.safe_load(open('${CONFIG_SOURCE}')); x=c['diffsbdd_collection']; assert x['expected_world_size']==8; assert x['expected_world_size']*x['num_shards_per_rank']*x['batch_size_per_rank']==x['fixed_attempts_total']==200000; assert c['model']['time_step']==500; assert (x['min_nodes'],x['max_nodes'],x['max_n_nodes'])==(10,30,30); assert x['reconstruction_method']=='openbabel'"

if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
  for required in \
    "${RUN_DIR}/config/resolved_config.yaml" \
    "${RUN_DIR}/source_snapshot/run.sh" \
    "${RUN_DIR}/source_snapshot/main_diffsbdd_collect_ddp.py" \
    "${RUN_DIR}/source_snapshot/validate_diffsbdd_trajectory_run.py" \
    "${RUN_DIR}/source_snapshot/ddpo" \
    "${RUN_DIR}/source_snapshot/DiffSBDD"; do
    if [[ ! -e "${required}" ]]; then
      echo "ERROR resume artifact missing: ${required}" >&2
      exit 1
    fi
  done
else
  if [[ -e "${RUN_DIR}" ]]; then
    echo "ERROR new run directory already exists: ${RUN_DIR}" >&2
    exit 1
  fi
fi

mkdir -p "${RUN_DIR}/config" "${RUN_DIR}/logs" "${RUN_DIR}/source_snapshot"
if [[ -z "${RESUME_RUN_DIR:-}" ]]; then
  cp "${CONFIG_SOURCE}" "${RUN_DIR}/config/resolved_config.yaml"
  cp "${RUN_SOURCE}" "${RUN_DIR}/source_snapshot/run.sh"
  cp "${ENTRY_SOURCE}" "${RUN_DIR}/source_snapshot/main_diffsbdd_collect_ddp.py"
  cp "${VALIDATOR_SOURCE}" "${RUN_DIR}/source_snapshot/validate_diffsbdd_trajectory_run.py"
  cp -a "${ROOT}/ddpo" "${RUN_DIR}/source_snapshot/ddpo"
  cp -a "${ROOT}/DiffSBDD" "${RUN_DIR}/source_snapshot/DiffSBDD"
  find "${RUN_DIR}/source_snapshot" -type d -name __pycache__ -prune -exec rm -rf {} +
  rm -rf "${RUN_DIR}/source_snapshot/DiffSBDD/checkpoints"
  ln -s "${ROOT}/DiffSBDD/checkpoints" \
    "${RUN_DIR}/source_snapshot/DiffSBDD/checkpoints"
  mkdir -p "${RUN_DIR}/source_snapshot/data"
  ln -s "${ROOT}/data/8GJL.pdb" \
    "${RUN_DIR}/source_snapshot/data/8GJL.pdb"

  git -C "${ROOT}" rev-parse HEAD > "${RUN_DIR}/config/git_commit.txt"
  git -C "${ROOT}" status --short > "${RUN_DIR}/config/git_status_short.txt"
  "${PYTHON_BIN}" -c "import sys,torch; print('python=',sys.version); print('torch=',torch.__version__); print('cuda=',torch.version.cuda)" > "${RUN_DIR}/config/environment.txt"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader > "${RUN_DIR}/config/gpu_preflight.csv"
  find "${RUN_DIR}/source_snapshot" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "${RUN_DIR}/config/source_sha256s.txt"
  sha256sum \
    "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
    "${ROOT}/data/8GJL.pdb" \
    "${RUN_DIR}/config/resolved_config.yaml" \
    > "${RUN_DIR}/config/input_sha256s.txt"
fi

env \
  DIFFB110_PROJECT_ROOT="${RUN_DIR}/source_snapshot" \
  PYTHONPATH="${RUN_DIR}/source_snapshot" \
  "${PYTHON_BIN}" -c \
  "from ddpo.integration.diffsbdd_model import DiffSBDDModel; from ddpo.runtime.trajectory_shard import TRAJECTORY_SHARD_FORMAT; print('snapshot_import_preflight=PASS', TRAJECTORY_SHARD_FORMAT)" \
  >> "${RUN_DIR}/logs/snapshot_import_preflight.log" 2>&1

printf '%s\n' "$(date -Is) resume=${RESUME_RUN_DIR:+true} run_dir=${RUN_DIR}" \
  >> "${RUN_DIR}/logs/start_history.log"

nohup env \
  DIFFB110_PROJECT_ROOT="${RUN_DIR}/source_snapshot" \
  CONFIG_PATH="${RUN_DIR}/config/resolved_config.yaml" \
  ENTRYPOINT="${RUN_DIR}/source_snapshot/main_diffsbdd_collect_ddp.py" \
  VALIDATOR="${RUN_DIR}/source_snapshot/validate_diffsbdd_trajectory_run.py" \
  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  bash "${RUN_DIR}/source_snapshot/run.sh" "${RUN_DIR}" \
  > "${RUN_DIR}/logs/launcher_stdout.log" 2>&1 &

echo "$!" > "${RUN_DIR}/launcher.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "PID=$!"
echo "RESUME_COMMAND=RESUME_RUN_DIR=${RUN_DIR} bash ${ROOT}/launch/start_diffsbdd_original_t500_8gpu_200k_trajectory_20260915.sh"
