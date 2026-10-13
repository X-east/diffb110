#!/usr/bin/env bash
# ESRPO 主线中间停留点全流程 pipeline:
#   生成 -> 取最大片段导出 relaxed SDF -> UniDock 对接
# 最后统一跑"五条标准(R1-R5)合理分子评估"并出曲线图.
#
# 每个 checkpoint 一个 step 目录, 支持断点续跑 (已完成的步骤自动跳过).
#
# 用法:
#   bash launch/run_esrpo_mainline_reasonable_pipeline.sh
# 覆盖变量:
#   MAINLINE_OUTPUT    训练输出根 (含 checkpoints/)
#   GEN_ROOT           生成+对接输出根
#   EVAL_OUT           合理分子评估输出目录
#   ATTEMPTS           每点生成分子数 (默认 1000)
#   GPU_ID             GPU (默认 0)
set -euo pipefail

PROJECT_ROOT="${DIFFB110_PROJECT_ROOT:-/home/md666/data/disk6/DiffB110}"
cd "${PROJECT_ROOT}"

MAINLINE_OUTPUT="${MAINLINE_OUTPUT:-OUT/diffsbdd_e2po_x_only_lr1e5_mainline}"
CKPT_DIR="${MAINLINE_OUTPUT}/checkpoints"
GEN_ROOT="${GEN_ROOT:-OUT/diffsbdd_e2po_mainline_intermediate_gen}"
EVAL_OUT="${EVAL_OUT:-analysis/e2po_mainline_reasonable_eval}"
ATTEMPTS="${ATTEMPTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-25}"
GPU_ID="${GPU_ID:-0}"

# 对接参数: 与 260901/260902 relaxed 对接口径完全一致
RECEPTOR="${RECEPTOR:-data/8gjl.pdbqt}"
CENTER_X=207.293
CENTER_Y=212.43
CENTER_Z=172.771
BOX_SIZE=20.0
UNIDOCK_BIN="${UNIDOCK_BIN:-/opt/unidock-1.1.3/bin/unidock}"

PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"
GEN_SCRIPT="launch/run_diffsbdd_global_esrpo_x_only_lr1e5_generation.sh"
EXPORT_SCRIPT="analysis/diffsbdd_global_e2po_structure_matched_x_only_lr1e5_full_generation_1000_260902/export_relaxed_sdf.py"
DOCK_SCRIPT="scripts/docking/dock_sdf_unidock.py"
EVAL_SCRIPT="tools/analysis_scripts/evaluate_reasonable_molecules.py"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR checkpoint dir not found: ${CKPT_DIR}" >&2
  exit 2
fi

# 收集 checkpoint (中间点升序, 终点 final 放最后)
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

echo "$(date -Is) INFO pipeline_start checkpoints=${#CKPTS[@]} attempts=${ATTEMPTS} gpu=${GPU_ID}"

EVAL_INPUTS=()
for ckpt in "${CKPTS[@]}"; do
  if [[ "${ckpt}" == *"_step="* ]]; then
    STEP_TAG="$(basename "${ckpt}" | sed -E 's/.*_step=([0-9]{6})\.pt/\1/')"
  else
    STEP_TAG="final"
  fi
  RUN_DIR="${GEN_ROOT}/step_${STEP_TAG}"
  MODEL_TAG="e2po_${STEP_TAG}"

  # ---- 1. 生成 ----
  if [[ ! -f "${RUN_DIR}/molecules.sdf" ]]; then
    echo "$(date -Is) INFO [${STEP_TAG}] generating molecules..."
    DPO_CHECKPOINT="${ckpt}" \
    RUN_DIR="${RUN_DIR}" \
    GPU_ID="${GPU_ID}" \
    ATTEMPTS="${ATTEMPTS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
      bash "${GEN_SCRIPT}"
  else
    echo "$(date -Is) SKIP [${STEP_TAG}] generation exists"
  fi

  # ---- 2. 导出 relaxed SDF (取最大片段 + sanitize) ----
  if [[ ! -f "${RUN_DIR}/molecules_relaxed_largest.sdf" ]]; then
    echo "$(date -Is) INFO [${STEP_TAG}] exporting relaxed SDF..."
    conda run --no-capture-output -n dif python "${EXPORT_SCRIPT}" \
      --sdf "${RUN_DIR}/molecules.sdf" \
      --out "${RUN_DIR}/molecules_relaxed_largest.sdf"
  else
    echo "$(date -Is) SKIP [${STEP_TAG}] relaxed SDF exists"
  fi

  # ---- 3. UniDock 对接 (relaxed 口径, 与 260902 参数一致) ----
  if [[ ! -f "${RUN_DIR}/docking_relaxed/docking_results.csv" ]]; then
    echo "$(date -Is) INFO [${STEP_TAG}] docking with UniDock..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run --no-capture-output -n dif python "${DOCK_SCRIPT}" \
      --model "${MODEL_TAG}" \
      --input-sdf "${RUN_DIR}/molecules_relaxed_largest.sdf" \
      --receptor "${RECEPTOR}" \
      --out-dir "${RUN_DIR}/docking_relaxed" \
      --unidock-bin "${UNIDOCK_BIN}" \
      --gpu-id "${GPU_ID}" \
      --batch-size 256 \
      --center-x "${CENTER_X}" --center-y "${CENTER_Y}" --center-z "${CENTER_Z}" \
      --size-x "${BOX_SIZE}" --size-y "${BOX_SIZE}" --size-z "${BOX_SIZE}" \
      --scoring vina --search-mode fast --exhaustiveness 8
  else
    echo "$(date -Is) SKIP [${STEP_TAG}] docking exists"
  fi

  EVAL_INPUTS+=("step_${STEP_TAG}:${RUN_DIR}/molecules.sdf:${RUN_DIR}/molecules_relaxed_largest.sdf:${RUN_DIR}/docking_relaxed/docking_results.csv:${ATTEMPTS}")
done

# 把最开始的基线 (Original, 未 DPO 训练) 也纳入评估, 作为曲线起点.
# 该 SDF 已由 260901 生成并完成 relaxed 对接.
BASELINE_RAW="${BASELINE_RAW:-analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901/molecules.sdf}"
BASELINE_RELAXED="${BASELINE_RELAXED:-analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901/molecules_relaxed_largest.sdf}"
BASELINE_DOCK="${BASELINE_DOCK:-analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901/docking_relaxed/docking_results.csv}"
if [[ -f "${BASELINE_RELAXED}" && -f "${BASELINE_DOCK}" ]]; then
  echo "$(date -Is) INFO including baseline Original in evaluation"
  EVAL_INPUTS+=("Original:${BASELINE_RAW}:${BASELINE_RELAXED}:${BASELINE_DOCK}:${ATTEMPTS}")
else
  echo "$(date -Is) WARN baseline Original inputs missing, skipped" >&2
fi

# ---- 4. 统一合理分子评估 (五条标准 R1-R5) + 出图 ----
echo "$(date -Is) INFO evaluating reasonable molecules (R1-R5) ..."
EVAL_ARGS=()
for item in "${EVAL_INPUTS[@]}"; do
  EVAL_ARGS+=("--input" "${item}")
done

conda run --no-capture-output -n dif python "${EVAL_SCRIPT}" \
  "${EVAL_ARGS[@]}" \
  --out-dir "${EVAL_OUT}"

echo "$(date -Is) INFO pipeline_completed eval_out=${EVAL_OUT}"
