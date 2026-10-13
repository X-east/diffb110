#!/usr/bin/env bash
set -euo pipefail
screen_gpu="${1:?Pass the available physical GPU index}"
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$task_dir/../.." && pwd)"
cd -- "$project_dir"
printf 'START %s gpu=%s environment=dif\n' "$(date -Is)" "$screen_gpu"
for stage in references batch2 esrpo; do
  case "$stage" in
    references) input="$task_dir/references_receptor_frame_input.sdf"; model=EPI_references ;;
    batch2) input="$task_dir/diffgui_batch2_input.sdf"; model=DiffGui_epoch80_batch2 ;;
    esrpo) input="$project_dir/OUT/diffsbdd_e2po_mainline_intermediate_gen/step_004000/molecules_relaxed_largest.sdf"; model=ESRPO_step4000 ;;
  esac
  printf 'STAGE_START %s %s\n' "$stage" "$(date -Is)"
  OMP_NUM_THREADS=18 MKL_NUM_THREADS=18 OPENBLAS_NUM_THREADS=18 conda run --no-capture-output -n dif python scripts/docking/dock_sdf_unidock.py \
    --model "$model" --input-sdf "$input" --receptor data/8gjl.pdbqt \
    --out-dir "$task_dir/docking_$stage" --gpu-id "$screen_gpu" \
    --center-x 207.293 --center-y 212.430 --center-z 172.771 \
    --size-x 20 --size-y 20 --size-z 20 --scoring vina --search-mode fast \
    --num-modes 1 --seed 42 --ligand-frame receptor_frame --cpu 18 --batch-size 128 --keep-split \
    > "$task_dir/docking_$stage.log" 2>&1
  printf 'STAGE_FINISH %s %s\n' "$stage" "$(date -Is)"
done
printf 'COMPLETE %s\n' "$(date -Is)"
printf 'complete\n' > "$task_dir/docking.status"
