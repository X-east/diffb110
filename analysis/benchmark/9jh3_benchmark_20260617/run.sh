#!/usr/bin/env bash
set -euo pipefail
# DiffSBDD + DiffGui 9JH3 generation: DiffSBDD→GPU0, DiffGui→GPU1, 各500 mol / 200 steps

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/md666/data/disk6/DiffB110"
DIFFGUI_ROOT="$REPO_ROOT/reference project/代码/DiffGui"
INPUTS="$ROOT/inputs"
OUTPUTS="$ROOT/outputs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/run_${TIMESTAMP}.log"

echo "============================================" | tee "$LOG"
echo "[9JH3] $(date '+%F %T')  start" | tee -a "$LOG"
echo "[9JH3] DiffSBDD → GPU 0  |  DiffGui → GPU 1" | tee -a "$LOG"
echo "[9JH3] 500 mols each  |  200 steps each" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# ── Step 1: Prepare inputs ──────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[9JH3] Step 1: prepare inputs" | tee -a "$LOG"
conda run -n dif python "$ROOT/scripts/prepare_inputs.py" \
    --source-pdb "$REPO_ROOT/data/9jh3-assembly1.pdb" \
    --out-dir "$INPUTS" \
    --ligand-chain R --ligand-resseq 401 --cutoff 8.0 \
    | tee -a "$LOG"
cat "$INPUTS/input_summary.json" | tee -a "$LOG"

# ── Step 2: DiffSBDD on GPU 0 ───────────────────────────────────
echo "" | tee -a "$LOG"
echo "[9JH3] Step 2: DiffSBDD (GPU 0)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 conda run -n dif python "$ROOT/scripts/run_diffsbdd.py" \
    --repo-root "$REPO_ROOT" \
    --checkpoint "$REPO_ROOT/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
    --pdbfile "$REPO_ROOT/data/9jh3-assembly1.pdb" \
    --ref-ligand "R:401" \
    --outfile "$OUTPUTS/diffsbdd_9jh3_200steps.sdf" \
    --n-samples 500 --batch-size 10 --timesteps 200 --seed 20260617 \
    2>&1 | tee -a "$LOG" &

# ── Step 3: DiffGui on GPU 1 ─────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[9JH3] Step 3: DiffGui (GPU 1)" | tee -a "$LOG"
(
    cd "$DIFFGUI_ROOT"
    CUDA_VISIBLE_DEVICES=1 \
    conda run -n diffgui python scripts/sample.py \
        --config "$ROOT/configs/diffgui_denovo.yml" \
        --outdir "$OUTPUTS/diffgui" \
        --device cuda:0 \
        >> "$LOG" 2>&1
) &

wait

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "[9JH3] ALL DONE $(date '+%F %T')" | tee -a "$LOG"
echo "[9JH3] DiffSBDD: $OUTPUTS/diffsbdd_9jh3_200steps.sdf" | tee -a "$LOG"
echo "[9JH3] DiffGui:  $OUTPUTS/diffgui/" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
