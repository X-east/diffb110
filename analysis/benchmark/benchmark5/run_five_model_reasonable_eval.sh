#!/usr/bin/env bash
# Compute R1-R5 "reasonable molecule" docking for the 5-model 1000-step 3-rep
# formal benchmark, using the SAME rule as E2PO (evaluate_reasonable_molecules.py).
# All 15 combinations (5 models x 3 reps) are evaluated in one call.
set -uo pipefail
cd /home/md666/data/disk6/DiffB110

S1=analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/filtered_sanitize_valid
S2=analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/filtered_sanitize_valid
S3=analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/filtered_sanitize_valid
DOCK=analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/docking
OUT=analysis/benchmark/benchmark5/five_model_1000steps_3rep_reasonable_eval

mkdir -p "$OUT"
declare -a INPUTS=()
for rep in 1 2 3; do
  case $rep in
    1) S=$S1;;
    2) S=$S2;;
    3) S=$S3;;
  esac
  for m in diffgui diffsbdd gcdm pmdm targetdiff; do
    if [ "$m" = gcdm ]; then dn=gcdm_sbdd; else dn=$m; fi
    sdf="$S/$m.sanitize_valid.sdf"
    csv="$DOCK/rep$rep/$dn/docking_results.csv"
    INPUTS+=(--input "$m-rep$rep:-:$sdf:$csv:1000")
  done
done

conda run -n dif python tools/analysis_scripts/evaluate_reasonable_molecules.py "${INPUTS[@]}" --out-dir "$OUT"
echo "EXIT=$?"
