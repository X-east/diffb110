#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/disk6/DiffB110"
RUN_DIR="${ROOT}/analysis/single_side_fragment_growth_20260718"
THREAD_ENV=(OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4)

mkdir -p "${RUN_DIR}/diffsbdd/mplconfig" "${RUN_DIR}/diffgui" "${RUN_DIR}/pmdm/input/frag_gen"

run_diffsbdd() {
  env CUDA_VISIBLE_DEVICES=1 "${THREAD_ENV[@]}" MPLCONFIGDIR="${RUN_DIR}/diffsbdd/mplconfig" \
    taskset -c 8-11 conda run -n dif python "${RUN_DIR}/run_with_seed.py" 2 \
    "${ROOT}/reference project/代码/DiffSBDD_upstream/inpaint.py" \
    "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
    --pdbfile "${RUN_DIR}/inputs/4dxd_honokiol_pocket10A.pdb" \
    --ref_ligand "${ROOT}/analysis/honokiol_9pc_docking_20260617/docking/successful_docked_outputs.sdf" \
    --fix_atoms "${RUN_DIR}/inputs/fixed_core_no_oh.sdf" \
    --center ligand \
    --n_samples 20 \
    --timesteps 100 \
    --resamplings 5 \
    --outfile "${RUN_DIR}/diffsbdd/diffsbdd_raw_20.sdf" \
    > "${RUN_DIR}/diffsbdd/generation.log" 2>&1
}

run_diffgui() {
  env -C "${ROOT}/reference project/代码/DiffGui" \
    CUDA_VISIBLE_DEVICES=2 "${THREAD_ENV[@]}" DIFFGUI_NUM_TIMESTEPS=100 \
    taskset -c 12-15 conda run -n diffgui python scripts/sample.py \
    --config "${RUN_DIR}/diffgui_sample.yaml" \
    --outdir "${RUN_DIR}/diffgui" \
    --device cuda:0 \
    --batch_size 1 \
    --seed 3 \
    > "${RUN_DIR}/diffgui/generation.log" 2>&1
}

run_pmdm() {
  env -C "${ROOT}/reference project/代码/modern_sbdd_20260531/PMDM" \
    CUDA_VISIBLE_DEVICES=3 "${THREAD_ENV[@]}" \
    taskset -c 16-19 conda run -n pmdm_modern_ref python \
    "${RUN_DIR}/run_pmdm_script.py" \
    "${ROOT}/reference project/代码/modern_sbdd_20260531/PMDM" sample_frag.py \
    --ckpt "${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/PMDM/500.pt" \
    --pdb_path "${RUN_DIR}/pmdm/input/4dxd_honokiol_pocket10A.pdb" \
    --mol_file "${RUN_DIR}/pmdm/input/fixed_core_no_oh.sdf" \
    --keep_index 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \
    --num_samples 20 \
    --batch_size 1 \
    --n_steps 100 \
    --sampling_type generalized \
    --seed 4 \
    -build_method reconstruct \
    > "${RUN_DIR}/pmdm/generation.log" 2>&1
}

run_diffsbdd &
pid_diffsbdd=$!
run_diffgui &
pid_diffgui=$!
run_pmdm &
pid_pmdm=$!

printf '%s\n' "diffsbdd=${pid_diffsbdd}" "diffgui=${pid_diffgui}" "pmdm=${pid_pmdm}" > "${RUN_DIR}/pids.txt"

status=0
wait "${pid_diffsbdd}" || status=1
wait "${pid_diffgui}" || status=1
wait "${pid_pmdm}" || status=1
exit "${status}"
