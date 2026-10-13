#!/usr/bin/env bash
# Step 2b: dock the sampled subset with UniDock, reusing the existing wrapper and
# the exact same protocol as the formal benchmarks:
#   receptor   data/8gjl.pdbqt
#   scoring    vina      search_mode fast      batch_size 128
set -uo pipefail

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
HERE="${HERE:-${ROOT}/analysis/reference_support_experiment}"
SUBSET="${SUBSET:-${HERE}/docking_subset}"
DOCK_SCRIPT="${DOCK_SCRIPT:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/dock_sdf_unidock.py}"
ENV_NAME="${ENV_NAME:-dif}"
RECEPTOR="${RECEPTOR:-${ROOT}/data/8gjl.pdbqt}"
UNIDOCK_BIN="${UNIDOCK_BIN:-/opt/unidock-1.1.3/bin/unidock}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
PARALLEL="${PARALLEL:-8}"

OUT_ROOT="${SUBSET}/docking"
mkdir -p "${OUT_ROOT}/logs"
MASTER_LOG="${OUT_ROOT}/master.log"
: >"${MASTER_LOG}"
log() { echo "[$(date +%T)] $*" | tee -a "${MASTER_LOG}"; }

mapfile -t SDFS < <(ls -1 "${SUBSET}"/sdf/part_*.sdf 2>/dev/null | sort)
if [[ ${#SDFS[@]} -eq 0 ]]; then
  log "ERROR: no part_*.sdf under ${SUBSET}/sdf — run select_docking_subset.py first"
  exit 1
fi

log "parts=${#SDFS[@]}  gpus=${GPU_IDS}  parallel=${PARALLEL}  batch_size=${BATCH_SIZE}"

run_one() {
  local i="$1"; local sdf="$2"
  # bash arrays cannot be exported, so re-split GPU_IDS inside the function
  local -a gpus; read -r -a gpus <<<"${GPU_IDS}"
  local n=${#gpus[@]}
  local gpu_id="${gpus[$((i % n))]}"
  local name; name="$(basename "${sdf}" .sdf)"
  local out_dir="${OUT_ROOT}/${name}"
  local log_path="${OUT_ROOT}/logs/${name}.log"
  local status_path="${OUT_ROOT}/logs/${name}.status"

  if [[ -s "${out_dir}/docking_results.csv" && "${FORCE:-0}" != "1" ]]; then
    echo ok >"${status_path}"; log "skip ${name}"; return 0
  fi

  if conda run -n "${ENV_NAME}" python "${DOCK_SCRIPT}" \
      --model "subset_${name}" \
      --input-sdf "${sdf}" \
      --receptor "${RECEPTOR}" \
      --out-dir "${out_dir}" \
      --unidock-bin "${UNIDOCK_BIN}" \
      --gpu-id "${gpu_id}" \
      --batch-size "${BATCH_SIZE}" \
      >"${log_path}" 2>&1; then
    echo ok >"${status_path}"; log "done ${name} (gpu ${gpu_id})"
  else
    echo failed >"${status_path}"; log "FAILED ${name} (gpu ${gpu_id}) see ${log_path}"; return 1
  fi
}

export -f run_one log
export ROOT HERE SUBSET DOCK_SCRIPT ENV_NAME RECEPTOR UNIDOCK_BIN BATCH_SIZE OUT_ROOT MASTER_LOG
export GPU_IDS PARALLEL FORCE

printf '%s\n' "${SDFS[@]}" | nl -ba -w1 -s' ' \
  | xargs -P "${PARALLEL}" -n 2 bash -c 'run_one "$0" "$1"' 2>>"${MASTER_LOG}"

ok=$(grep -l ok "${OUT_ROOT}"/logs/*.status 2>/dev/null | wc -l)
fail=$(grep -l failed "${OUT_ROOT}"/logs/*.status 2>/dev/null | wc -l)
log "completed: ok=${ok} failed=${fail} / ${#SDFS[@]}"

# merge all per-part docking_results.csv into one table
conda run -n "${ENV_NAME}" python - "${OUT_ROOT}" <<'PY' 2>>"${MASTER_LOG}"
import csv, sys
from pathlib import Path
root = Path(sys.argv[1])
files = sorted(root.glob("part_*/docking_results.csv"))
if not files:
    print("no results to merge"); sys.exit(0)
rows, fields = [], None
for f in files:
    with f.open() as fh:
        r = csv.DictReader(fh)
        if fields is None:
            fields = ["part"] + list(r.fieldnames or [])
        for row in r:
            rows.append({"part": f.parent.name, **row})
out = root / "docking_results_all.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
print(f"merged {len(rows):,} rows from {len(files)} parts -> {out}")
PY

log "ALL DONE"
