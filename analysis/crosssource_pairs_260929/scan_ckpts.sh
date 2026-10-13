#!/usr/bin/env bash
# Checkpoint sweep: generate 250 attempts per DPO checkpoint on GPU 5 so the
# "structural gain vs molecular simplification" trade-off can be plotted
# against training steps.
set -uo pipefail
ROOT=/data/disk6/DiffB110
CKPT_DIR=$ROOT/OUT/diffsbdd_esrpo_crosssource_ring_smoke/checkpoints
SCAN=$ROOT/OUT/crosssource_ring_scan
mkdir -p "$SCAN"

for step in 000400 000800 001200 final; do
  if [[ "$step" == "final" ]]; then
    CKPT=$CKPT_DIR/diffsbdd-global-dpo-epoch=0001.pt
    TAG=final
  else
    CKPT=$CKPT_DIR/diffsbdd-global-dpo-epoch=0001_step=${step}.pt
    TAG=step${step}
  fi
  RUN_DIR=$SCAN/$TAG
  echo "$(date -Is) SCAN start $TAG"
  DPO_CHECKPOINT=$CKPT GPU_ID=5 ATTEMPTS=250 RUN_DIR=$RUN_DIR \
    PATH=/home/md666/miniforge3/bin:$PATH \
    bash $ROOT/launch/run_diffsbdd_global_esrpo_x_only_lr1e5_generation.sh \
    > $SCAN/gen_${TAG}.log 2>&1
  echo "$(date -Is) SCAN done  $TAG rc=$?"
done
echo "$(date -Is) SCAN ALL DONE"
