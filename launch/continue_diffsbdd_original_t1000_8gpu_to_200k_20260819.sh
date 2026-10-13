#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/disk6/DiffB110"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/diffsbdd_collect/diffsbdd-original-t1000-8gpu-100k-trajectory_20260730_194643}"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"
CONFIG_SOURCE="${ROOT}/launch/diffsbdd_original_t1000_8gpu_200k_continuation_20260819.yaml"
CONFIG_PATH="${RUN_DIR}/config/resolved_config_200k_continuation_20260819.yaml"
ENTRYPOINT="${RUN_DIR}/source_snapshot/main_diffsbdd_collect_ddp.py"
VALIDATOR="${RUN_DIR}/source_snapshot/validate_diffsbdd_trajectory_run.py"
MASTER_PORT="${MASTER_PORT:-29504}"

for command_name in bash cp flock setsid sha256sum; do
  command -v "${command_name}" >/dev/null || { echo "ERROR missing ${command_name}" >&2; exit 1; }
done
for required in "${PYTHON_BIN}" "${CONFIG_SOURCE}" "${ENTRYPOINT}" "${VALIDATOR}"; do
  [[ -e "${required}" ]] || { echo "ERROR missing ${required}" >&2; exit 1; }
done
bash -n "$0"
for rank in 00 01 02 03 04 05 06 07; do
  manifest="${RUN_DIR}/manifests/rank_${rank}.jsonl"
  [[ -s "${manifest}" ]] || { echo "ERROR missing manifest ${manifest}" >&2; exit 1; }
  count="$(${PYTHON_BIN} -c 'import json,sys; p,r=sys.argv[1:]; rows=[json.loads(x) for x in open(p) if x.strip()]; assert len(rows)>=1250, len(rows); assert all(int(x.get("shard_index",-1))==i and int(x.get("rank",-1))==int(r) for i,x in enumerate(rows)); print(len(rows))' "${manifest}" "${rank}")" || { echo "ERROR invalid manifest ${manifest}" >&2; exit 1; }
  echo "INFO rank ${rank} has ${count} contiguous completed shards; resume will append to shard 2499"
done

if [[ -e "${CONFIG_PATH}" ]]; then
  cmp -s "${CONFIG_SOURCE}" "${CONFIG_PATH}" || { echo "ERROR continuation config snapshot differs" >&2; exit 1; }
else
  cp "${CONFIG_SOURCE}" "${CONFIG_PATH}"
fi
if [[ ! -e "${RUN_DIR}/summary_100k_original.json" ]]; then
  cp "${RUN_DIR}/summary.json" "${RUN_DIR}/summary_100k_original.json"
  cp "${RUN_DIR}/completion_report.json" "${RUN_DIR}/completion_report_100k_original.json"
fi

"${PYTHON_BIN}" -c "import yaml; c=yaml.safe_load(open('${CONFIG_PATH}')); x=c['diffsbdd_collection']; assert (x['expected_world_size'],x['batch_size_per_rank'],x['num_shards_per_rank'],x['fixed_attempts_total'])==(8,10,2500,200000); assert c['model']['time_step']==1000; assert (x['min_nodes'],x['max_nodes'],x['max_n_nodes'])==(10,30,30)"
env DIFFB110_PROJECT_ROOT="${RUN_DIR}/source_snapshot" PYTHONPATH="${RUN_DIR}/source_snapshot" "${PYTHON_BIN}" -c "from ddpo.integration.diffsbdd_model import DiffSBDDModel; from ddpo.runtime.trajectory_shard import TRAJECTORY_SHARD_FORMAT; print('snapshot_import_preflight=PASS', TRAJECTORY_SHARD_FORMAT)" >> "${RUN_DIR}/logs/continuation_preflight.log" 2>&1

exec > >(tee -a "${RUN_DIR}/logs/continuation_200k.launch.log") 2>&1
exec 9> "${RUN_DIR}/continuation_200k.launch.lock"
flock -n 9 || { echo "ERROR another continuation launcher is active" >&2; exit 1; }
printf '%s\n' "$(date -Is) continuation=100k_to_200k config=${CONFIG_PATH} shards_per_rank=2500 seeds=1-8 restored_from_current_manifests=true" >> "${RUN_DIR}/logs/start_history.log"

echo "$(date -Is) INFO starting RNG-continuous resume: append each rank through shard 2499"
setsid env DIFFB110_PROJECT_ROOT="${RUN_DIR}/source_snapshot" PYTHONPATH="${RUN_DIR}/source_snapshot" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0" PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" "${ENTRYPOINT}" --config "${CONFIG_PATH}" --output-dir "${RUN_DIR}" --resume --log-level INFO &
TORCHRUN_PID="$!"
wait "${TORCHRUN_PID}"

"${PYTHON_BIN}" "${VALIDATOR}" --run-dir "${RUN_DIR}" --expected-ranks 8 --expected-shards-per-rank 2500 --attempts-per-shard 10 --timesteps 1000 --max-n-nodes 30 > "${RUN_DIR}/logs/completion_validation_200k.log" 2>&1
echo "$(date -Is) INFO continuation complete summary=${RUN_DIR}/summary.json"
