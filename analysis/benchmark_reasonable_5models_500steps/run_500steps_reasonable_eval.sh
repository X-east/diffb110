#!/usr/bin/env bash
# 五模型 500 步横评的"合理分子子集"评估 (R1-R5 全清口径).
#
# 与论文表 2.1/3.2 (T=500, 3 次重复) 口径对齐:
#   relaxed SDF = five_model_repeats_500steps_260826/rep{4,5,6}_*/filtered_sanitize_valid/<model>.sanitize_valid.sdf
#   docking CSV = five_model_500steps_full_eval_260903/docking/rep{4,5,6}/<slug>/docking_results.csv
# 路径来源: benchmark5/five_model_500steps_full_eval_260903/run_500steps_docking.sh
# 复用 tools/analysis_scripts/evaluate_reasonable_molecules.py (与 E2PO 同一判据).
set -euo pipefail

ROOT=/home/md666/data/disk6/DiffB110
cd "${ROOT}"

R=analysis/benchmark/benchmark5/five_model_repeats_500steps_260826
D=analysis/benchmark/benchmark5/five_model_500steps_full_eval_260903/docking
PY=/home/md666/miniforge3/envs/dif/bin/python
OUT=analysis/benchmark_reasonable_5models_500steps
mkdir -p "${OUT}"

# model|sanitize_valid 文件名|docking 目录 slug
MODELS=(
  "DiffSBDD|diffsbdd.sanitize_valid.sdf|diffsbdd"
  "DiffGui|diffgui.sanitize_valid.sdf|diffgui"
  "PMDM|pmdm.sanitize_valid.sdf|pmdm"
  "GCDM-SBDD|gcdm.sanitize_valid.sdf|gcdm_sbdd"
  "TargetDiff|targetdiff.sanitize_valid.sdf|targetdiff"
)

INPUTS=()
for rep in 4 5 6; do
  if [[ "${rep}" -eq 4 ]]; then sub=rep4_seed22_29
  elif [[ "${rep}" -eq 5 ]]; then sub=rep5_seed30_37
  else sub=rep6_seed38_45
  fi
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name file slug <<<"${entry}"
    sdf="${R}/${sub}/filtered_sanitize_valid/${file}"
    dock="${D}/rep${rep}/${slug}/docking_results.csv"
    if [[ ! -f "${sdf}" ]]; then echo "ERROR missing sdf: ${sdf}" >&2; exit 2; fi
    if [[ ! -f "${dock}" ]]; then echo "ERROR missing dock: ${dock}" >&2; exit 2; fi
    INPUTS+=(--input "${name}_rep${rep}:-:${sdf}:${dock}:1000")
  done
done

echo "[run] evaluating 15 specs (5 models x 3 repeats, 500 steps) -> ${OUT}"
"${PY}" tools/analysis_scripts/evaluate_reasonable_molecules.py "${INPUTS[@]}" --out-dir "${OUT}"
echo "[run] done. csv=${OUT}/reasonable_molecules_metrics.csv"
