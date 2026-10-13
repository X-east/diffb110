#!/usr/bin/env bash
# 三臂 ESRPO 消融并行提交: B / C / D 各占一张 GPU。
#
# 卡分配原则: 把最多 pair（update 最多）的臂放到负载最低的卡。
#   C: 72,000 train pairs -> 36,000 updates -> GPU 7 (负载最低)
#   D: 65,433 train pairs -> 32,717 updates -> GPU 6
#   B: 56,628 train pairs -> 28,314 updates -> GPU 5
#   （分配依据: 2026-09-15 11:4x 的 nvidia-smi: 7=17%, 6=36%, 0-5=91-92%）
#
# 用法:
#   bash launch/run_esrpo_frag_bcd_3gpu_20260915.sh            # 提交三臂
#   bash launch/run_esrpo_frag_bcd_3gpu_20260915.sh B          # 只提交某一臂
#   GPU_B=5 GPU_C=7 GPU_D=6 bash launch/...                    # 覆盖卡分配
#
# 每个臂:
#   - 单卡（ESRPO mainline 实测 updates = train_pairs / batch_pairs，无数据并行）
#   - --disable-generation: 训练完只出 checkpoint，生成/对接走后续 pipeline
#   - 已完成（存在 completion.json）自动跳过
set -uo pipefail

PROJECT_ROOT="${DIFFB110_PROJECT_ROOT:-/home/md666/data/disk6/DiffB110}"
PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"
ENTRY="${PROJECT_ROOT}/ddpo/main_diffsbdd_global_dpo.py"

GPU_B="${GPU_B:-5}"
GPU_C="${GPU_C:-7}"
GPU_D="${GPU_D:-6}"

ONLY="${1:-}"

if [[ ! -f "${ENTRY}" ]]; then echo "ERROR missing entrypoint: ${ENTRY}" >&2; exit 2; fi
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "ERROR missing python: ${PYTHON_BIN}" >&2; exit 2; fi

launch_arm() {
  local arm="$1" gpu="$2"
  local cfg="${PROJECT_ROOT}/launch/e2po_frag_bcd_${arm}_20260915.yaml"
  local out="${PROJECT_ROOT}/analysis/e2po_frag_bcd_${arm}_20260915"
  local log="${out}/train_console.log"

  if [[ ! -f "${cfg}" ]]; then echo "SKIP ${arm}: missing config ${cfg}" >&2; return 1; fi
  if [[ -f "${out}/completion.json" ]]; then echo "SKIP ${arm}: already completed (${out}/completion.json)"; return 0; fi
  local ntrain
  ntrain=$(wc -l < "${PROJECT_ROOT}/analysis/reference_support_experiment/pairs_bcd/${arm}/train_pairs.csv")
  mkdir -p "${out}"

  echo "$(date -Is) LAUNCH arm=${arm} gpu=${gpu} train_csv_rows=${ntrain} out=${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/DiffSBDD" \
  PYTHONUNBUFFERED=1 \
  nohup "${PYTHON_BIN}" -u "${ENTRY}" \
      --config "${cfg}" \
      --variant x_only \
      --epochs 1 \
      --device cuda:0 \
      --disable-generation \
      --output-dir "${out}" \
      > "${log}" 2>&1 &
  echo "$!" > "${out}/launcher.pid"
  echo "        pid=$! log=${log}"
}

if [[ -n "${ONLY}" ]]; then
  case "${ONLY}" in
    B) launch_arm B "${GPU_B}" ;;
    C) launch_arm C "${GPU_C}" ;;
    D) launch_arm D "${GPU_D}" ;;
    *) echo "ERROR unknown arm: ${ONLY} (expect B|C|D)" >&2; exit 2 ;;
  esac
else
  launch_arm C "${GPU_C}"
  launch_arm D "${GPU_D}"
  launch_arm B "${GPU_B}"
fi

echo
echo "已提交的臂会在各自 analysis/e2po_frag_bcd_<ARM>_20260915/train_console.log 里输出。"
echo "查看进度: tail -f analysis/e2po_frag_bcd_{B,C,D}_20260915/train_console.log"
