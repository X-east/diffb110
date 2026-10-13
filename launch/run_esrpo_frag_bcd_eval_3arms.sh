#!/usr/bin/env bash
# 三臂 ESRPO 统一口径评估: 生成 -> relaxed -> 对接 -> R1-R5 评估 -> frag/对接对照
#
# 目的: 判断 B(对接标签) / C(真实性标签) / D(组合标签) 哪一臂真的更好。
# 协议与 mainline 完全一致 (launch/run_esrpo_mainline_reasonable_pipeline.sh)，
# 唯一额外要求是 **三臂必须用同一个 seed**，否则生成样本不可比。
#
# 每个 (arm, step) 的产出:
#   OUT/e2po_frag_bcd_eval/<ARM>/step_<STEP>/molecules.sdf
#                                            /molecules_relaxed_largest.sdf
#                                            /docking_relaxed/docking_results.csv
# 最后汇总:
#   analysis/e2po_frag_bcd_compare/frag_arm_comparison.csv   (对接 + soft_novel)
#   analysis/e2po_frag_bcd_compare/<evaluate_reasonable_molecules 的产物> (R1-R5)
#
# 用法:
#   bash launch/run_esrpo_frag_bcd_eval_3arms.sh                  # 全部 checkpoint
#   STEPS="002000 004000 010000 final" bash launch/...           # 只跑指定停留点
#   ARMS="D" bash launch/...                                     # 只跑一臂
#   GPU_ID=0 GPU_DOCK=0 bash launch/...                          # 指定卡
#
# 断点续跑: 任一已有产物自动跳过（生成脚本本身拒绝覆盖，故先判断再调用）。
set -uo pipefail

PROJECT_ROOT="${DIFFB110_PROJECT_ROOT:-/home/md666/data/disk6/DiffB110}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"
GEN_SCRIPT="launch/run_diffsbdd_global_esrpo_x_only_lr1e5_generation.sh"
EXPORT_SCRIPT="analysis/diffsbdd_global_e2po_structure_matched_x_only_lr1e5_full_generation_1000_260902/export_relaxed_sdf.py"
DOCK_SCRIPT="scripts/docking/dock_sdf_unidock.py"
EVAL_SCRIPT="tools/analysis_scripts/evaluate_reasonable_molecules.py"
CMP_SCRIPT="tools/analysis_scripts/frag_arm_comparison.py"
UNIDOCK_BIN="${UNIDOCK_BIN:-/opt/unidock-1.1.3/bin/unidock}"

ARMS="${ARMS:-B C D}"
GPU_ID="${GPU_ID:-0}"                 # 生成用卡
GPU_DOCK="${GPU_DOCK:-${GPU_ID}}"      # 对接用卡
ATTEMPTS="${ATTEMPTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-25}"
SEED="${SEED:-1}"                     # ★ 三臂共用，保证生成样本可比
STEPS="${STEPS:-}"                    # 空 = 全部 checkpoint；否则空格分隔的 step tag

GEN_ROOT="${GEN_ROOT:-OUT/e2po_frag_bcd_eval}"
EVAL_OUT="${EVAL_OUT:-analysis/e2po_frag_bcd_compare}"
CMP_OUT="${CMP_OUT:-${EVAL_OUT}}"

RECEPTOR="${RECEPTOR:-data/8gjl.pdbqt}"
CENTER_X=207.293; CENTER_Y=212.43; CENTER_Z=172.771; BOX_SIZE=20.0

for f in "${GEN_SCRIPT}" "${EXPORT_SCRIPT}" "${DOCK_SCRIPT}" "${EVAL_SCRIPT}" \
         "${CMP_SCRIPT}" "${RECEPTOR}" "${PYTHON_BIN}"; do
  if [[ ! -e "${f}" ]]; then echo "ERROR missing required file: ${f}" >&2; exit 2; fi
done

mkdir -p "${EVAL_OUT}" "${CMP_OUT}"
echo "$(date -Is) INFO eval_start arms='${ARMS}' gpu_gen=${GPU_ID} gpu_dock=${GPU_DOCK} seed=${SEED} attempts=${ATTEMPTS}"

EVAL_INPUTS=()
CMP_INPUTS=()

for arm in ${ARMS}; do
  CKPT_DIR="${PROJECT_ROOT}/analysis/e2po_frag_bcd_${arm}_20260915/checkpoints"
  if [[ ! -d "${CKPT_DIR}" ]]; then echo "SKIP arm ${arm}: no checkpoints at ${CKPT_DIR}" >&2; continue; fi

  mapfile -t CKPTS < <(
    ls "${CKPT_DIR}" | grep -E 'diffsbdd-global-dpo-epoch=0001(_step=[0-9]{6})?\.pt$' \
      | sed -E 's/.*epoch=0001(_step=([0-9]{6}))?\.pt/\2/' \
      | awk -v d="${CKPT_DIR}/" '{
          s = ($1 == "" ? 999999 : $1 + 0);
          printf "%08d\t%s\n", s, d (s == 999999 ? "diffsbdd-global-dpo-epoch=0001.pt" : "diffsbdd-global-dpo-epoch=0001_step=" $1 ".pt")
        }' | sort -k1,1n | cut -f2-
  )
  if [[ ${#CKPTS[@]} -eq 0 ]]; then echo "SKIP arm ${arm}: no checkpoints matched" >&2; continue; fi
  echo "$(date -Is) INFO arm ${arm}: ${#CKPTS[@]} checkpoints"

  for ckpt in "${CKPTS[@]}"; do
    if [[ "${ckpt}" == *"_step="* ]]; then
      STEP_TAG="$(basename "${ckpt}" | sed -E 's/.*_step=([0-9]{6})\.pt/\1/')"
    else
      STEP_TAG="final"
    fi
    if [[ -n "${STEPS}" ]] && [[ " ${STEPS} " != *" ${STEP_TAG} "* ]]; then
      continue
    fi

    RUN_DIR="${GEN_ROOT}/arm_${arm}/step_${STEP_TAG}"
    MODEL_TAG="e2po_bcd${arm}_${STEP_TAG}"
    mkdir -p "${RUN_DIR}"

    # ---- 1. 生成 ----
    if [[ ! -f "${RUN_DIR}/molecules.sdf" ]]; then
      echo "$(date -Is) INFO [${arm}/${STEP_TAG}] generating..."
      DPO_CHECKPOINT="${ckpt}" RUN_DIR="${RUN_DIR}" GPU_ID="${GPU_ID}" \
      SEED="${SEED}" ATTEMPTS="${ATTEMPTS}" BATCH_SIZE="${BATCH_SIZE}" \
        bash "${GEN_SCRIPT}" || { echo "ERROR generation failed [${arm}/${STEP_TAG}]" >&2; continue; }
    else
      echo "$(date -Is) SKIP [${arm}/${STEP_TAG}] generation exists"
    fi

    # ---- 2. relaxed SDF (最大片段 + sanitize) ----
    if [[ ! -f "${RUN_DIR}/molecules_relaxed_largest.sdf" ]]; then
      echo "$(date -Is) INFO [${arm}/${STEP_TAG}] exporting relaxed SDF..."
      conda run --no-capture-output -n dif python "${EXPORT_SCRIPT}" \
        --sdf "${RUN_DIR}/molecules.sdf" \
        --out "${RUN_DIR}/molecules_relaxed_largest.sdf" \
        || { echo "ERROR relax failed [${arm}/${STEP_TAG}]" >&2; continue; }
    else
      echo "$(date -Is) SKIP [${arm}/${STEP_TAG}] relaxed SDF exists"
    fi

    # ---- 3. UniDock 对接（与 mainline 参数完全一致）----
    if [[ ! -f "${RUN_DIR}/docking_relaxed/docking_results.csv" ]]; then
      echo "$(date -Is) INFO [${arm}/${STEP_TAG}] docking..."
      CUDA_VISIBLE_DEVICES="${GPU_DOCK}" conda run --no-capture-output -n dif python "${DOCK_SCRIPT}" \
        --model "${MODEL_TAG}" \
        --input-sdf "${RUN_DIR}/molecules_relaxed_largest.sdf" \
        --receptor "${RECEPTOR}" \
        --out-dir "${RUN_DIR}/docking_relaxed" \
        --unidock-bin "${UNIDOCK_BIN}" \
        --gpu-id "${GPU_DOCK}" \
        --batch-size 256 \
        --center-x "${CENTER_X}" --center-y "${CENTER_Y}" --center-z "${CENTER_Z}" \
        --size-x "${BOX_SIZE}" --size-y "${BOX_SIZE}" --size-z "${BOX_SIZE}" \
        --scoring vina --search-mode fast --exhaustiveness 8 \
        || { echo "ERROR docking failed [${arm}/${STEP_TAG}]" >&2; continue; }
    else
      echo "$(date -Is) SKIP [${arm}/${STEP_TAG}] docking exists"
    fi

    EVAL_INPUTS+=("${arm}_step_${STEP_TAG}:${RUN_DIR}/molecules.sdf:${RUN_DIR}/molecules_relaxed_largest.sdf:${RUN_DIR}/docking_relaxed/docking_results.csv:${ATTEMPTS}")
    CMP_INPUTS+=("${arm}_step_${STEP_TAG}:${RUN_DIR}/molecules_relaxed_largest.sdf:${RUN_DIR}/docking_relaxed/docking_results.csv")
  done
done

# ---- 基线 Original（未 DPO 训练）作为曲线起点 ----
BASELINE_RAW="analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901/molecules.sdf"
BASELINE_RELAXED="analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901/molecules_relaxed_largest.sdf"
BASELINE_DOCK="analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901/docking_relaxed/docking_results.csv"
if [[ -f "${BASELINE_RELAXED}" && -f "${BASELINE_DOCK}" ]]; then
  echo "$(date -Is) INFO including baseline Original"
  EVAL_INPUTS+=("Original:${BASELINE_RAW}:${BASELINE_RELAXED}:${BASELINE_DOCK}:${ATTEMPTS}")
  CMP_INPUTS+=("Original_final:${BASELINE_RELAXED}:${BASELINE_DOCK}")
else
  echo "$(date -Is) WARN baseline Original missing, skipped" >&2
fi

if [[ ${#EVAL_INPUTS[@]} -eq 0 ]]; then echo "ERROR nothing to evaluate" >&2; exit 1; fi

# ---- 4. R1-R5 合理分子评估 ----
echo "$(date -Is) INFO running reasonable-molecule evaluation on ${#EVAL_INPUTS[@]} inputs ..."
EVAL_ARGS=()
for item in "${EVAL_INPUTS[@]}"; do EVAL_ARGS+=("--input" "${item}"); done
conda run --no-capture-output -n dif python "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}" --out-dir "${EVAL_OUT}" \
  || echo "WARN reasonable-molecule evaluation returned non-zero" >&2

# ---- 5. frag 真实性 + 对接离群值对照 ----
echo "$(date -Is) INFO running fragment-realism / docking comparison ..."
CMP_ARGS=()
for item in "${CMP_INPUTS[@]}"; do CMP_ARGS+=("--input" "${item}"); done
"${PYTHON_BIN}" "${CMP_SCRIPT}" "${CMP_ARGS[@]}" --out-dir "${CMP_OUT}" \
  || echo "WARN frag comparison returned non-zero" >&2

echo "$(date -Is) INFO eval_completed out=${EVAL_OUT}"
