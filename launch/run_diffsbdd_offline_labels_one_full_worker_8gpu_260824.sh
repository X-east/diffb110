#!/usr/bin/env bash
set -euo pipefail

# One independent process per GPU. Each process prepares its complete modulo
# shard assignment and invokes UniDock once with one ligand_index file.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_ROOT="${OFFLINE_ROOT:-/data/disk6/DiffB110-offline-iql}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/launch/diffsbdd_offline_labels_alidiff_unidock_affinity_8gpu_full_260824.yaml}"
INDEX_DIR="${INDEX_DIR:-${DATA_ROOT}/analysis/offline_iql_full_index_rebuilt_260824_v7}"
INDEX_PATH="${INDEX_PATH:-${INDEX_DIR}/episode_index.jsonl}"
INDEX_SUMMARY="${INDEX_SUMMARY:-${INDEX_DIR}/dataset_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/analysis/offline_iql_labels_alidiff_unidock_affinity_8gpu_full_260824}"
PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESUME="${RESUME:-1}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-200000}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-20000}"

WORKER_SCRIPT="${PROJECT_ROOT}/scripts/offline_rl/label_trajectory_shards_batched.py"
VERIFY_INDEX_SCRIPT="${PROJECT_ROOT}/scripts/offline_rl/verify_episode_index.py"
FINALIZER="${OFFLINE_ROOT}/scripts/offline_rl/label_trajectory_shards.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" || ! -f "${WORKER_SCRIPT}" || ! -f "${FINALIZER}" ]]; then
  echo "Missing config or labeling entrypoint" >&2
  echo "config=${CONFIG_PATH}" >&2
  echo "worker_script=${WORKER_SCRIPT}" >&2
  echo "finalizer=${FINALIZER}" >&2
  exit 2
fi
if [[ "${NUM_WORKERS}" =~ [^0-9] || "${NUM_WORKERS}" -le 0 ]]; then
  echo "NUM_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "${PRECHECK_ONLY}" != "0" && "${PRECHECK_ONLY}" != "1" ]]; then
  echo "PRECHECK_ONLY must be 0 or 1" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne "${NUM_WORKERS}" ]]; then
  echo "GPU_IDS count (${#GPU_ARRAY[@]}) must equal NUM_WORKERS (${NUM_WORKERS})" >&2
  exit 2
fi
for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
  GPU_ARRAY[worker_id]="${GPU_ARRAY[worker_id]//[[:space:]]/}"
  if [[ ! "${GPU_ARRAY[worker_id]}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: ${GPU_ARRAY[worker_id]}" >&2
    exit 2
  fi
  for ((prior = 0; prior < worker_id; prior++)); do
    if [[ "${GPU_ARRAY[worker_id]}" == "${GPU_ARRAY[prior]}" ]]; then
      echo "Duplicate GPU id: ${GPU_ARRAY[worker_id]}" >&2
      exit 2
    fi
  done
done

if [[ -n "${PYTHONPATH:-}" ]]; then
  LABEL_PYTHONPATH="${OFFLINE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"
else
  LABEL_PYTHONPATH="${OFFLINE_ROOT}:${PROJECT_ROOT}"
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/config"
exec > >(tee -a "${OUTPUT_DIR}/logs/full_launcher.log") 2>&1

echo "$(date --iso-8601=seconds) full_worker_label_launch_start"
echo "project_root=${PROJECT_ROOT}"
echo "offline_root=${OFFLINE_ROOT}"
echo "config=${CONFIG_PATH}"
echo "index=${INDEX_PATH}"
echo "output=${OUTPUT_DIR}"
echo "workers=${NUM_WORKERS} gpu_ids=${GPU_ARRAY[*]}"
echo "python=${PYTHON_BIN} pythonpath=${LABEL_PYTHONPATH} resume=${RESUME}"
echo "submission_policy=one_full_worker_chunk_per_gpu; unidock_internal_split=true"
echo "git_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
git -C "${PROJECT_ROOT}" status --short -- \
  scripts/offline_rl/label_trajectory_shards_batched.py \
  scripts/offline_rl/verify_episode_index.py \
  launch/diffsbdd_offline_labels_alidiff_unidock_affinity_8gpu_full_260824.yaml \
  launch/run_diffsbdd_offline_labels_one_full_worker_8gpu_260824.sh

snapshot_file() {
  local source_path="$1"
  local snapshot_path="$2"
  if [[ -f "${snapshot_path}" ]]; then
    if ! cmp -s "${source_path}" "${snapshot_path}"; then
      echo "Run-local snapshot differs: ${snapshot_path}" >&2
      exit 2
    fi
  else
    cp "${source_path}" "${snapshot_path}"
  fi
}

snapshot_file "${CONFIG_PATH}" "${OUTPUT_DIR}/config/resolved_config.yaml"
snapshot_file "${WORKER_SCRIPT}" "${OUTPUT_DIR}/config/label_trajectory_shards_batched.py"
snapshot_file "${VERIFY_INDEX_SCRIPT}" "${OUTPUT_DIR}/config/verify_episode_index.py"
snapshot_file "${BASH_SOURCE[0]}" "${OUTPUT_DIR}/config/launcher_snapshot.sh"

echo "$(date --iso-8601=seconds) index_verification_start"
"${PYTHON_BIN}" "${VERIFY_INDEX_SCRIPT}" \
  --index "${INDEX_PATH}" \
  --summary "${INDEX_SUMMARY}" \
  --expected-episodes "${EXPECTED_EPISODES}" \
  --expected-shards "${EXPECTED_SHARDS}"
echo "$(date --iso-8601=seconds) index_verification_passed"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader || echo "WARNING nvidia-smi query failed; continuing to PyTorch preflight"
else
  echo "WARNING nvidia-smi is unavailable"
fi
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 8; x=torch.ones(1, device="cuda"); torch.cuda.synchronize(); print("cuda_preflight=PASS device_count=" + str(torch.cuda.device_count()) + " names=" + ",".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))'

COMMON_ARGS=(
  --config "${CONFIG_PATH}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --num-workers "${NUM_WORKERS}"
)

echo "$(date --iso-8601=seconds) worker_preflight_start"
declare -a PREFLIGHT_PIDS=()
for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
  gpu_id="${GPU_ARRAY[worker_id]}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHONPATH="${LABEL_PYTHONPATH}" \
    "${PYTHON_BIN}" "${WORKER_SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --worker-id "${worker_id}" \
    --gpu-id "${gpu_id}" \
    --preflight-only \
    > "${OUTPUT_DIR}/logs/preflight_worker_${worker_id}.log" 2>&1 &
  PREFLIGHT_PIDS+=("$!")
done
preflight_failed=0
for pid in "${PREFLIGHT_PIDS[@]}"; do
  if ! wait "${pid}"; then
    preflight_failed=1
  fi
done
if [[ "${preflight_failed}" -ne 0 ]]; then
  echo "Worker preflight failed; inspect ${OUTPUT_DIR}/logs/preflight_worker_*.log" >&2
  exit 1
fi
echo "$(date --iso-8601=seconds) worker_preflight_passed"
if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "$(date --iso-8601=seconds) precheck_only_complete"
  exit 0
fi

declare -a WORKER_PIDS=()
cleanup_workers() {
  local pid
  for pid in "${WORKER_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_workers INT TERM

echo "$(date --iso-8601=seconds) worker_launch_start"
: > "${OUTPUT_DIR}/worker_pids.tsv"
for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
  gpu_id="${GPU_ARRAY[worker_id]}"
  worker_args=(
    "${COMMON_ARGS[@]}"
    --worker-id "${worker_id}"
    --gpu-id "${gpu_id}"
  )
  if [[ "${RESUME}" == "1" ]]; then
    worker_args+=(--resume)
  fi
  worker_log="${OUTPUT_DIR}/logs/worker_${worker_id}.launcher.log"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHONPATH="${LABEL_PYTHONPATH}" \
    setsid "${PYTHON_BIN}" "${WORKER_SCRIPT}" \
    "${worker_args[@]}" \
    > "${worker_log}" 2>&1 < /dev/null &
  worker_pid="$!"
  WORKER_PIDS+=("${worker_pid}")
  printf 'worker_id=%s\tgpu_id=%s\tpid=%s\tlog=%s\n' \
    "${worker_id}" "${gpu_id}" "${worker_pid}" "${worker_log}" \
    >> "${OUTPUT_DIR}/worker_pids.tsv"
  echo "worker=${worker_id} gpu=${gpu_id} pid=${worker_pid} log=${worker_log} started"
done

failed=0
for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
  if wait "${WORKER_PIDS[worker_id]}"; then
    echo "worker=${worker_id} completed"
  else
    echo "worker=${worker_id} failed; see ${OUTPUT_DIR}/logs/worker_${worker_id}.launcher.log" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "$(date --iso-8601=seconds) worker_stage_failed" >&2
  exit 1
fi

echo "$(date --iso-8601=seconds) finalization_start"
PYTHONPATH="${OFFLINE_ROOT}" \
  "${PYTHON_BIN}" "${FINALIZER}" \
  --config "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --worker-id 0 \
  --num-workers 1 \
  --finalize-only

"${PYTHON_BIN}" - "${OUTPUT_DIR}/summary.json" "${EXPECTED_EPISODES}" <<'PY'
import json
import sys

summary = json.loads(open(sys.argv[1], encoding="utf-8").read())
expected = int(sys.argv[2])
if summary.get("scope") != "full_episode_index":
    raise SystemExit(f"unexpected final scope: {summary.get('scope')}")
if int(summary["counts"]["attempts"]) != expected:
    raise SystemExit(
        f"attempt denominator mismatch: {summary['counts']['attempts']} != {expected}"
    )
print(
    "final_summary_verified="
    + json.dumps(
        {
            "scope": summary["scope"],
            "attempts": summary["counts"]["attempts"],
            "rdkit_valid": summary["counts"]["rdkit_valid"],
            "docking_success": summary["counts"]["docking_success"],
            "actual_good": summary["counts"]["actual_good"],
        },
        sort_keys=True,
    )
)
PY

touch "${OUTPUT_DIR}/COMPLETE"
echo "$(date --iso-8601=seconds) full_worker_label_launch_complete"
