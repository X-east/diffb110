#!/usr/bin/env bash
# DiffSBDD 保留线最终主线 ESRPO 生成脚本 (structure-matched, x-only, lr=1e-5).
# 主线方向见 docs/project_context.md / AGENTS.md. 具体超参待优化.
#
# 用法:
#   DPO_CHECKPOINT=... GPU_ID=0 ATTEMPTS=1000 bash launch/run_diffsbdd_global_esrpo_x_only_lr1e5_generation.sh
# 所有关键路径/数量均可通过环境变量覆盖 (见下方 :- 默认值).
set -euo pipefail

PROJECT_ROOT="${DIFFB110_PROJECT_ROOT:-/home/md666/data/disk6/DiffB110}"

# 默认 DPO checkpoint 指向 ESRPO 主线训练产出 (lr1e5_full_260901 验证来源).
# 注意: 之前 run_..._generation_1000_260901.sh 误指向 test_260901 (md5 不同), 此处已修正.
DPO_CHECKPOINT="${DPO_CHECKPOINT:-${PROJECT_ROOT}/analysis/diffsbdd_global_e2po_structure_matched_x_only_lr1e5_full_260901/checkpoints/diffsbdd-global-dpo-epoch=0001.pt}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT_ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt}"

# 输出目录: 默认落在 OUT/ 下, 与主线训练 output_dir 同根. 用时间戳避免覆盖.
RUN_STAMP="${RUN_STAMP:-$(date +%y%m%d)}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/OUT/diffsbdd_e2po_x_only_lr1e5_generation_${RUN_STAMP}}"
GEN_CHECKPOINT="${RUN_DIR}/global_dpo_generation.ckpt"
SDF_PATH="${RUN_DIR}/molecules.sdf"

# GPU / 种子: 按规则 seed = gpu_id (1-based). GPU0 -> seed 1.
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-$((GPU_ID + 1))}"
PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"
ATTEMPTS="${ATTEMPTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-25}"

mkdir -p "${RUN_DIR}"
if [[ ! -f "${DPO_CHECKPOINT}" || ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "ERROR missing checkpoint input" >&2
  exit 2
fi
if [[ -e "${SDF_PATH}" || -e "${GEN_CHECKPOINT}" ]]; then
  echo "ERROR generation output already exists: ${RUN_DIR}" >&2
  exit 2
fi

echo "$(date -Is) INFO generation_start launcher_pid=$$ physical_gpu=${GPU_ID} seed=${SEED} attempts=${ATTEMPTS} batch_size=${BATCH_SIZE}"
echo "$(date -Is) INFO dpo_checkpoint=${DPO_CHECKPOINT}"
echo "$(date -Is) INFO output_sdf=${SDF_PATH}"

conda run --no-capture-output -n dif python "${PROJECT_ROOT}/scripts/export_diffsbdd_global_dpo_checkpoint.py" \
  --dpo-checkpoint "${DPO_CHECKPOINT}" \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --output "${GEN_CHECKPOINT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/DiffSBDD${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export BABEL_LIBDIR="${BABEL_LIBDIR:-/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0}"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/DiffSBDD/generate_ligands.py" "${GEN_CHECKPOINT}" \
  --pdbfile "${PROJECT_ROOT}/data/8GJL.pdb" \
  --ref_ligand "C:1101" \
  --outfile "${SDF_PATH}" \
  --n_samples "${ATTEMPTS}" \
  --batch_size "${BATCH_SIZE}" \
  --timesteps 500 \
  --seed "${SEED}" \
  --resamplings 10 \
  --jump_length 1 \
  --reconstruction_method openbabel \
  --all_frags

echo "$(date -Is) INFO generation_completed attempts=${ATTEMPTS} output=${SDF_PATH}"
