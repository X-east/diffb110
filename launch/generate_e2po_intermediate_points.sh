#!/usr/bin/env bash
# E2PO 主线中间停留点批量生成 driver.
# 遍历 OUT/diffsbdd_e2po_x_only_lr1e5_mainline/checkpoints/ 下所有
#   diffsbdd-global-dpo-epoch=0001_step=NNNNNN.pt  (训练中途快照)
#   diffsbdd-global-dpo-epoch=0001.pt              (终点, 记为 step=final)
# 对每个 checkpoint 调用 run_diffsbdd_global_e2po_x_only_lr1e5_generation.sh 生成分子.
#
# 用法:
#   bash launch/generate_e2po_intermediate_points.sh
# 覆盖变量:
#   MAINLINE_OUTPUT  训练输出根 (默认 OUT/diffsbdd_e2po_x_only_lr1e5_mainline)
#   GEN_ROOT         生成输出根 (默认 OUT/diffsbdd_e2po_mainline_intermediate_gen)
#   ATTEMPTS_PER_POINT 每点分子数 (默认 1000, 与 260902-02 口径一致)
#   BATCH_SIZE       生成批大小 (默认 25)
#   GPU_ID           单卡 GPU (默认 0; seed=gpu_id+1=1)
set -euo pipefail

PROJECT_ROOT="${DIFFB110_PROJECT_ROOT:-/home/md666/data/disk6/DiffB110}"
MAINLINE_OUTPUT="${MAINLINE_OUTPUT:-${PROJECT_ROOT}/OUT/diffsbdd_e2po_x_only_lr1e5_mainline}"
CKPT_DIR="${MAINLINE_OUTPUT}/checkpoints"
GEN_ROOT="${GEN_ROOT:-${PROJECT_ROOT}/OUT/diffsbdd_e2po_mainline_intermediate_gen}"
ATTEMPTS_PER_POINT="${ATTEMPTS_PER_POINT:-1000}"
BATCH_SIZE="${BATCH_SIZE:-25}"
GPU_ID="${GPU_ID:-0}"

GEN_SCRIPT="${PROJECT_ROOT}/launch/run_diffsbdd_global_e2po_x_only_lr1e5_generation.sh"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR checkpoint dir not found: ${CKPT_DIR}" >&2
  echo "      请先完成 mainline 重训 (launch/diffsbdd_global_e2po_x_only_lr1e5_mainline.yaml)." >&2
  exit 2
fi

# 收集所有中间点 + 终点, 按 step 数值排序 (终点放最后).
# 用 ls + sed 解析, 避免 rg.
mapfile -t CKPTS < <(
  ls "${CKPT_DIR}" | grep -E 'diffsbdd-global-dpo-epoch=0001(_step=[0-9]{6})?\.pt$' \
    | sed -E 's/.*epoch=0001(_step=([0-9]{6}))?\.pt/\2/' \
    | awk -v d="${CKPT_DIR}/" '{
        step = ($1 == "" ? 999999 : $1+0);
        printf "%08d\t%s\n", step, d (step==999999 ? "diffsbdd-global-dpo-epoch=0001.pt" : "diffsbdd-global-dpo-epoch=0001_step=" $1 ".pt")
      }' | sort -k1,1n | cut -f2-
)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "ERROR no checkpoints matched in ${CKPT_DIR}" >&2
  exit 2
fi

echo "$(date -Is) INFO found ${#CKPTS[@]} checkpoints to generate"
for ckpt in "${CKPTS[@]}"; do
  # 从文件名解析 step 标签: step=0002000 -> 0002000, 终点 -> final
  if [[ "${ckpt}" == *"_step="* ]]; then
    STEP_TAG="$(basename "${ckpt}" | sed -E 's/.*_step=([0-9]{6})\.pt/\1/')"
  else
    STEP_TAG="final"
  fi
  RUN_DIR="${GEN_ROOT}/step_${STEP_TAG}"
  if [[ -e "${RUN_DIR}/molecules.sdf" ]]; then
    echo "$(date -Is) SKIP already generated: ${RUN_DIR}"
    continue
  fi
  echo "$(date -Is) INFO generating step=${STEP_TAG} ckpt=${ckpt}"
  DPO_CHECKPOINT="${ckpt}" \
  RUN_DIR="${RUN_DIR}" \
  GPU_ID="${GPU_ID}" \
  ATTEMPTS="${ATTEMPTS_PER_POINT}" \
  BATCH_SIZE="${BATCH_SIZE}" \
    bash "${GEN_SCRIPT}"
done

echo "$(date -Is) INFO all intermediate point generations done -> ${GEN_ROOT}"
