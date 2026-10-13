#!/usr/bin/env bash
# 8GJL/C1101 5-model benchmark: 1000 mol/model, T=500, 7 GPUs (1-7)
# Models: DiffGui, DiffSBDD, PMDM, GCDM-SBDD, TargetDiff
# Policy: "Generate 1000 attempts, stop" — validity rate is a model metric.
set -euo pipefail

# Initialize conda for non-interactive shells (nohup). Without this, `conda run`
# fails with "condabin/conda: Permission denied": the conda shell function is
# undefined in non-interactive bash, and condabin/conda lacks the executable bit.
source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="/home/md666/data/disk6/DiffB110"
EXP="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps"
INPUTS="${EXP}/inputs"
SCRIPTS="${EXP}/scripts"
TIMESTEPS=500
COUNTS=(143 143 143 143 143 143 142)   # 6×143 + 1×142 = 1000
GPUS=(1 2 3 4 5 6 7)

mkdir -p "${EXP}/logs" "${EXP}/outputs" "${EXP}/aggregated" "${EXP}/evaluation"

# ---------------------------------------------------------------------------
# Helper: run N shards in parallel across GPUs, wait for all
# ---------------------------------------------------------------------------
run_parallel() {
  local model_name="$1"
  local log_dir="${EXP}/logs/${model_name}"
  mkdir -p "${log_dir}"
  shift

  local pids=()
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export PYTHONDONTWRITEBYTECODE=1
      "$@" "${gpu_id}" "${count}"
    ) > "${log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    echo "[${model_name}] launched gpu=${gpu_id} count=${count} pid=${pids[-1]}"
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[${model_name}] gpu=${GPUS[$idx]} completed"
    else
      echo "[${model_name}] gpu=${GPUS[$idx]} FAILED (see ${log_dir}/gpu_${GPUS[$idx]}.log)"
      failed=1
    fi
  done
  return "${failed}"
}

# ---------------------------------------------------------------------------
# 1. DiffGui  (conda env: diffgui, --fixed_attempts)
# ---------------------------------------------------------------------------
prepare_diffgui_configs() {
  local cfg_dir="${EXP}/configs/diffgui"
  mkdir -p "${cfg_dir}"
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
    cat > "${cfg_dir}/diffgui_c1101_gpu${gpu_id}.yml" <<EOF
model:
  checkpoint: ckpt/trained.pt
  target: ${INPUTS}/8GJL_C1101_pocket8A.pdb
  ligand: None
  frag: None
  gen_mode: denovo
  logp: 2.0
  tpsa: 100
  sa: 1.0
  qed: 0.8
  aff: 12.0
bond_predictor: ckpt/bond_trained.pt
sample:
  seed: $gpu_id
  batch_size: 2
  num_mols: ${count}
  num_timesteps: ${TIMESTEPS}
  save_traj_prob: 0.0
  sample: true
  sample_method: priori
  mode: pocket
  test_id: 0
  add_edge: None
  gui_strength: 3.0
  guidance:
  - uncertainty
  - 0.0001
data:
  name: protein_ligand
  path: data/PDBbind_v2020_pocket10
  split: data/PDBbind_pocket10_split.pt
  protein_root: data/PDBbind_v2020
  dataset: pdbbind
  transform:
    ligand_atom_mode: aromatic
    random_rot: false
    sample: false
EOF
  done
}

run_diffgui_one() {
  local gpu_id="$1"
  local count="$2"
  local project="${ROOT}/reference project/代码/DiffGui"
  local out_dir="${EXP}/outputs/diffgui/gpu_${gpu_id}"
  local cfg="${EXP}/configs/diffgui/diffgui_c1101_gpu${gpu_id}.yml"
  local mpl_dir="${EXP}/mplconfig/diffgui/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  cd "${project}"
  export DIFFGUI_NUM_TIMESTEPS="${TIMESTEPS}"
  export PYTHONPATH="${EXP}/runtime_patch/diffgui:${project}${PYTHONPATH:+:${PYTHONPATH}}"
  export MPLCONFIGDIR="${mpl_dir}"
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  conda run -n diffgui python scripts/sample.py \
    --config "${cfg}" \
    --outdir "${out_dir}" \
    --device cuda:0 \
    --batch_size 2 \
    --fixed_attempts \
    --seed "${gpu_id}"
}

aggregate_diffgui() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffGui \
    --output "${EXP}/aggregated/diffgui.sdf" \
    --limit 1000 \
    "${EXP}/outputs/diffgui"
}

# ---------------------------------------------------------------------------
# 2. DiffSBDD  (conda env: dif, already "generate N stop")
# ---------------------------------------------------------------------------
run_diffsbdd_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/diffsbdd/gpu_${gpu_id}"
  mkdir -p "${out_dir}"
  conda run -n dif python "${SCRIPTS}/run_diffsbdd_shard.py" \
    --repo-root "${ROOT}" \
    --checkpoint "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
    --pdbfile "${ROOT}/data/8GJL.pdb" \
    --ref-ligand "C:1101" \
    --outfile "${out_dir}/generated.sdf" \
    --n-samples "${count}" \
    --batch-size 10 \
    --timesteps "${TIMESTEPS}" \
    --seed "$gpu_id" \
    --reconstruction-method openbabel
}

aggregate_diffsbdd() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffSBDD \
    --output "${EXP}/aggregated/diffsbdd.sdf" \
    --limit 1000 \
    "${EXP}/outputs/diffsbdd"
}

# ---------------------------------------------------------------------------
# 3. PMDM  (conda env: pmdm_modern_ref, --fixed-attempts)
# ---------------------------------------------------------------------------
run_pmdm_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/pmdm/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/pmdm/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  export MPLCONFIGDIR="${mpl_dir}"
  conda run -n pmdm_modern_ref python \
    "${ROOT}/analysis/benchmark/modern_sbdd_20260531/run_pmdm_for_pdb.py" \
    --repo "${ROOT}/reference project/代码/modern_sbdd_20260531/PMDM" \
    --ckpt "${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/PMDM/500.pt" \
    --pdb-path "${INPUTS}/8GJL_C1101_pocket8A.pdb" \
    --outdir "${out_dir}" \
    --num-atom 24 \
    --num-samples "${count}" \
    --batch-size 32 \
    --n-steps "${TIMESTEPS}" \
    --device cuda \
    --seed "$gpu_id" \
    --fixed-attempts
}

aggregate_pmdm() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model PMDM \
    --output "${EXP}/aggregated/pmdm.sdf" \
    --limit 1000 \
    "${EXP}/outputs/pmdm"
}

# ---------------------------------------------------------------------------
# 4. GCDM-SBDD  (conda env: gcdm_sbdd_modern_ref, already "generate N stop")
# ---------------------------------------------------------------------------
run_gcdm_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/gcdm/gpu_${gpu_id}"
  mkdir -p "${out_dir}"
  cd "${ROOT}/reference project/代码/modern_sbdd_20260531/GCDM-SBDD"
  conda run -n gcdm_sbdd_modern_ref python generate_ligands.py \
    "${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/GCDM-SBDD/extracted/checkpoints/crossdocked_ca_joint_egnn.ckpt" \
    --pdbfile "${ROOT}/data/8GJL.pdb" \
    --ref-ligand "C:1101" \
    --outdir "${out_dir}" \
    --n_samples "${count}" \
    --timesteps "${TIMESTEPS}" \
    --seed "${gpu_id}"
}

aggregate_gcdm() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model GCDM-SBDD \
    --output "${EXP}/aggregated/gcdm.sdf" \
    --limit 1000 \
    "${EXP}/outputs/gcdm"
}

# ---------------------------------------------------------------------------
# 5. TargetDiff  (conda env: targetdiff, already "generate N stop")
# ---------------------------------------------------------------------------
prepare_targetdiff_configs() {
  local cfg_dir="${EXP}/configs/targetdiff"
  mkdir -p "${cfg_dir}"
  conda run -n dif python "${SCRIPTS}/prepare_targetdiff_configs.py" \
    --out-dir "${cfg_dir}" \
    --result-root "${EXP}/outputs/targetdiff" \
    --protein "${INPUTS}/8GJL_C1101_protein_only.pdb" \
    --ligand "${INPUTS}/8GJL_C1101_ligand.sdf" \
    --checkpoint "/home/md666/data/disk6/XDC/targetdiff/pretrained_models/pretrained_diffusion.pt" \
    --counts "143,143,143,143,143,143,142" \
    --base-seed 1
  # NOTE: prepare_targetdiff_configs.py uses base_seed + gpu_id*101; the
  # generated yml seeds are then overwritten in-place to seed=gpu_id to
  # align with DiffGui/DiffSBDD/PMDM/GCDM-SBDD (see Random Seed Norm).
}

run_targetdiff_one() {
  local gpu_id="$1"
  local count="$2"  # unused, count is in the config
  local idx=$((gpu_id - 1))
  local project="/home/md666/data/disk6/XDC/targetdiff"
  local cfg="${EXP}/configs/targetdiff/targetdiff_c1101_gpu${idx}.yml"
  cd "${project}"
  conda run -n targetdiff python scripts/sample_from_protein_ligand.py \
    "${cfg}"
}

aggregate_targetdiff() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model TargetDiff \
    --output "${EXP}/aggregated/targetdiff.sdf" \
    --limit 1000 \
    "${EXP}/outputs/targetdiff"
}

# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
evaluate_all() {
  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${EXP}/evaluation" \
    --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" \
    --sdf "DiffSBDD=${EXP}/aggregated/diffsbdd.sdf" \
    --sdf "PMDM=${EXP}/aggregated/pmdm.sdf" \
    --sdf "GCDM-SBDD=${EXP}/aggregated/gcdm.sdf" \
    --sdf "TargetDiff=${EXP}/aggregated/targetdiff.sdf"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "[benchmark] 8GJL/C1101 5-model benchmark started at $(date -Is)"
echo "[benchmark] target=8GJL/C1101  timesteps=${TIMESTEPS}  total_per_model=1000  gpus=${GPUS[*]}"

# DiffGui
prepare_diffgui_configs
run_parallel diffgui run_diffgui_one
aggregate_diffgui

# DiffSBDD
run_parallel diffsbdd run_diffsbdd_one
aggregate_diffsbdd

# PMDM
run_parallel pmdm run_pmdm_one
aggregate_pmdm

# GCDM-SBDD
run_parallel gcdm run_gcdm_one
aggregate_gcdm

# TargetDiff
prepare_targetdiff_configs
run_parallel targetdiff run_targetdiff_one
aggregate_targetdiff

# Evaluate all
evaluate_all

echo "[benchmark] finished at $(date -Is)"
