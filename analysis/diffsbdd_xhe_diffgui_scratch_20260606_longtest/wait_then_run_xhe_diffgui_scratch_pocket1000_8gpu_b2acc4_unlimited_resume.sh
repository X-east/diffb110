#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest"
RUN_DIR="${RUN_ROOT}/lightning/diffsbdd-xhe-diffgui-scratch-pocket1000-8gpu-260606-b2acc4-8ep"
SOURCE_CKPT="${RUN_DIR}/checkpoints/last.ckpt"
RESUME_CKPT="${RUN_DIR}/checkpoints/last_continue_unlimited.ckpt"
UNLIMITED_SCRIPT="${RUN_ROOT}/run_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_unlimited_resume.sh"
LOGGER_VERSION="metrics_resume_unlimited"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_PIDS="${WAIT_PIDS:-3985496 3985598 3985599 3985600 3985601 3985602 3985603 3985604}"

read -r -a wait_pids <<< "${WAIT_PIDS}"
echo "[xhe-diffgui-scratch-unlimited-watcher] Waiting for current to16 run."
echo "[xhe-diffgui-scratch-unlimited-watcher] Watched PIDs: ${wait_pids[*]}"

while true; do
  live_pids=()
  for pid in "${wait_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      live_pids+=("${pid}")
    fi
  done

  if [[ "${#live_pids[@]}" -eq 0 ]]; then
    break
  fi

  echo "[xhe-diffgui-scratch-unlimited-watcher] Still running: ${live_pids[*]}"
  wait_pids=("${live_pids[@]}")
  sleep "${POLL_SECONDS}"
done

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "[xhe-diffgui-scratch-unlimited-watcher] Missing source checkpoint: ${SOURCE_CKPT}" >&2
  exit 1
fi

cp "${SOURCE_CKPT}" "${RESUME_CKPT}"

conda run --no-capture-output -n dif python - "${RESUME_CKPT}" "${LOGGER_VERSION}" <<'PY'
import sys

import torch

path, logger_version = sys.argv[1], sys.argv[2]
ckpt = torch.load(path, map_location="cpu")
hp = ckpt.setdefault("hyper_parameters", {})
hp["n_epochs"] = -1
hp["checkpoint_every_n_epochs"] = 5

local_logging = hp.get("local_logging")
if hasattr(local_logging, "__dict__"):
    local_logging = dict(vars(local_logging))
elif local_logging is None:
    local_logging = {}
else:
    local_logging = dict(local_logging)

local_logging.update(
    {
        "backend": "csv",
        "version": logger_version,
        "log_every_n_steps": 50,
        "print_train_every_n_batches": 250,
    }
)
hp["local_logging"] = local_logging
torch.save(ckpt, path)
print(
    "[xhe-diffgui-scratch-unlimited-watcher] Patched checkpoint "
    f"{path}: n_epochs={hp['n_epochs']}, "
    f"checkpoint_every_n_epochs={hp['checkpoint_every_n_epochs']}, "
    f"logger={local_logging['version']}"
)
PY

echo "[xhe-diffgui-scratch-unlimited-watcher] Starting unlimited resume."
bash "${UNLIMITED_SCRIPT}"
