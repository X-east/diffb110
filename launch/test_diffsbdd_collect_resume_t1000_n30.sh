#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"
CONFIG="${ROOT}/launch/smoke_diffsbdd_collect_resume_t1000_n30.yaml"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/diffsbdd_collect/smoke_diffsbdd_collect_resume_t1000_n30_$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29505}"
FIRST_PID=""
SECOND_PID=""
COLLECTOR_PID=""

cleanup() {
  local exit_code=$?
  for process_id in "${FIRST_PID}" "${SECOND_PID}"; do
    if [[ "${process_id}" =~ ^[0-9]+$ ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill -TERM -- "-${process_id}" 2>/dev/null || true
    fi
  done
  exit "${exit_code}"
}
trap cleanup EXIT

mkdir -p "${RUN_DIR}/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export PYTHONUNBUFFERED=1

launch_collector() {
  local log_path="$1"
  setsid "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    "${ROOT}/ddpo/main_diffsbdd_collect_ddp.py" \
      --config "${CONFIG}" \
      --output-dir "${RUN_DIR}" \
      --resume \
      --log-level INFO \
    > "${log_path}" 2>&1 &
  COLLECTOR_PID="$!"
}

launch_collector "${RUN_DIR}/logs/first_interrupted_attempt.log"
FIRST_PID="${COLLECTOR_PID}"
deadline=$((SECONDS + 180))
while true; do
  rank0_rows=0
  rank1_rows=0
  [[ -f "${RUN_DIR}/manifests/rank_00.jsonl" ]] && rank0_rows=$(wc -l < "${RUN_DIR}/manifests/rank_00.jsonl")
  [[ -f "${RUN_DIR}/manifests/rank_01.jsonl" ]] && rank1_rows=$(wc -l < "${RUN_DIR}/manifests/rank_01.jsonl")
  if [[ "${rank0_rows}" -ge 1 && "${rank1_rows}" -ge 1 ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "ERROR timed out waiting for first completed shards" >&2
    exit 1
  fi
  sleep 1
done

kill -TERM -- "-${FIRST_PID}"
set +e
wait "${FIRST_PID}"
first_exit=$?
set -e
FIRST_PID=""
if [[ "${first_exit}" -eq 0 ]]; then
  echo "ERROR first attempt completed before interruption" >&2
  exit 1
fi

for rank in 00 01; do
  rows=$(wc -l < "${RUN_DIR}/manifests/rank_${rank}.jsonl")
  if [[ "${rows}" -ne 1 ]]; then
    echo "ERROR rank ${rank} expected exactly one pre-resume shard, got ${rows}" >&2
    exit 1
  fi
done
sha256sum \
  "${RUN_DIR}/shards/rank_00/shard_0000.pt" \
  "${RUN_DIR}/shards/rank_01/shard_0000.pt" \
  > "${RUN_DIR}/logs/pre_resume_shard_sha256s.txt"

launch_collector "${RUN_DIR}/logs/resumed_attempt.log"
SECOND_PID="${COLLECTOR_PID}"
wait "${SECOND_PID}"
SECOND_PID=""

(cd "${ROOT}" && "${PYTHON_BIN}" scripts/validate_diffsbdd_trajectory_run.py \
  --run-dir "${RUN_DIR}" \
  --expected-ranks 2 \
  --expected-shards-per-rank 2 \
  --attempts-per-shard 10 \
  --timesteps 1000 \
  --max-n-nodes 30) \
  > "${RUN_DIR}/logs/completion_validation.log"

(cd / && sha256sum -c "${RUN_DIR}/logs/pre_resume_shard_sha256s.txt") \
  > "${RUN_DIR}/logs/pre_resume_hash_check.log"
"${PYTHON_BIN}" -c "import json,pathlib; p=pathlib.Path('${RUN_DIR}'); s=json.loads((p/'summary.json').read_text()); assert s['attempts_total']==40; assert s['attempts_new']==20; assert s['completed_shards_total']==4; print('resume_contract=PASS attempts_total=40 attempts_new=20 shard0_hashes_unchanged=true')" \
  | tee "${RUN_DIR}/logs/resume_contract.log"

echo "RUN_DIR=${RUN_DIR}"
