# Workflow Commands

> **最后更新**: 2026-09-29

This document records the current verified workflow entrypoints, launch scripts, and supporting utilities in the repository.

## 0. Current Project Workflow

The DiffSBDD retained route has a confirmed mainline: **ESRPO** (endpoint structural-rule
preference optimization; structure-matched, x-only, lr 1e-5) - see section 5.
Its **ChEMBL version** (cross-source preference pairs, winners taken from real ChEMBL
molecules) is in progress - also section 5.

Other routes remain parallel work packages / comparisons:

```text
cross-model diffusion benchmark evidence
  -> ESRPO mainline (DiffSBDD endpoints)  |  ESRPO-ChEMBL version (ChEMBL winners)
  -> parallel: BC/AWR/IQL, AliDiff-style DPO, DecompDPO-style DPO, online DDPO, DiffGui RL
  -> independent generation from each checkpoint
  -> screen and rerank generated molecules
  -> hand off top candidates to molecular dynamics simulation
```

Pipeline status / code entry points: `docs/管线总览.md`.

For Part I reports and execution records, use fixed sampling attempts and fixed
seed rules. Do not refill invalid generations to force every model to have the
same number of written SDF molecules.

## 1. Environments

### Main DiffB110 / DiffSBDD environment

Use `dif` for the main project runtime, DiffSBDD DDPO, reward scoring, analysis scripts, classifier training, and most tests.

```bash
conda activate dif
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
```

### DiffGui environment

Use `diffgui` for any task that imports the reference DiffGui model, loads `reference project/代码/DiffGui/ckpt/trained.pt`, performs DiffGui generation, or runs DiffGui PPO/DDPO.

Do not install DiffGui-only dependencies into `dif`.

## 2. DiffSBDD Offline IQL Parallel Branch

DiffSBDD offline IQL is one parallel branch in the `dif` environment. The main repository provides complete-trajectory collection; any IQL actor training remains conditional on the data, critic, numerical-stability, and independent-generation checks documented for that branch. Do not describe IQL as the globally confirmed mainline.

### Available collection entrypoint

```bash
conda run -n dif python ddpo/main_diffsbdd_collect_ddp.py --help
```

### Launch scripts

| Command | Use |
|---|---|
| `conda run -n dif python ddpo/main_diffsbdd_collect_ddp.py --help` | Inspect the DiffSBDD complete-trajectory collector |

For an IQL comparison, follow the branch record in `work_records/26-7/260731/260731-01 DiffSBDD去噪轨迹离线IQL支线全过程记录.md`: freeze the relevant episodes/manifests, build finite reward sidecars, audit actual-good/scaffold support, freeze the split, run matched controls, perform held-out critic diagnostics, then independently evaluate fresh molecules.

### Related parallel route evidence

- `runs/diffgui-*`
- `launch/archive/logs/diffgui_ring_control_matrix_*`
- `analysis/diffgui_rl/`
- `ddpo/diffgui/runtime/rollout.py`
- `ddpo/diffgui/runtime/logprob_adapter.py`
- `ddpo/diffgui/runtime/actor.py`
- `ddpo/diffgui/runtime/training_filter.py`
- `ddpo/diffgui/runtime/good_pool.py`
- `ddpo/diffgui/runtime/replay.py`

`launch/archive/logs/` and `launch/archive/runtime_logs/` are run evidence, not
maintained entrypoints.

The IQL branch gates and control design are documented in:

- `work_records/26-7/260731/260731-01 DiffSBDD去噪轨迹离线IQL支线全过程记录.md`

## 3. Cross-Route Generation, Screening, And MD Handoff

After any route checkpoint is selected for evaluation, the downstream work should be recorded as independent generation and screening, not inferred from training loss alone.

### Generation From Route Checkpoints

- Use the `dif` environment for DiffSBDD checkpoints.
- Save generated SDFs, run logs, resolved config snapshots, seed rules, and
  attempt counts under `OUT/` or a dated `analysis/` directory.
- Keep the benchmark seed convention unless a run explicitly documents another
  reason: `seed = gpu_id` for sharded generation and `seed = 1` for single-GPU
  generation.

### Screening And Reranking

Screen generated molecules with a fixed funnel:

1. RDKit readable and sanitize-valid.
2. Canonical SMILES deduplication.
3. UniDock RE-CmeB docking under the documented box and receptor.
4. Ligand efficiency to control molecule-size bias.
5. QED, SA, LogP, TPSA, MW, HBA/HBD, and related drug-like descriptors.
6. Ring/pathology facts, fragments, bad bond lengths, abnormal valence, and
   chemistry-guard fields.
7. ChEMBL-like MLP/D-MPNN or fingerprint-prior rescoring.
8. Manual 3D interaction review for top candidates.

### MD Candidate Handoff

Molecular dynamics is a downstream validation stage. Before starting MD, create
or update a task-specific record with:

- selected molecule IDs and source SDF paths;
- docking pose or pose-generation source;
- protein structure, protonation, ligand preparation, and force-field choices;
- MD engine, simulation length, replica count, solvent/ion setup, and acceptance
  criteria;
- final output directory under `analysis/` or a dedicated MD record directory.

## 4. Online PPO/DDPO Parallel And Historical Paths

The online DiffSBDD and DiffGui PPO/DDPO paths may be used as parallel experiments, comparisons, or historical reproductions according to their run records. Do not mix their checkpoints or trajectory semantics with DiffSBDD offline branches, and do not substitute recorded behavior scores for offline-IQL old-policy log probabilities.

### Main Python entrypoint

```bash
torchrun --nproc_per_node=<N> ddpo/main_diffsbdd_ddp.py --config <launch/config.yaml>
```

### Standard launch scripts

| Command | Use |
|---|---|
| `bash launch/smoke.sh` | Standard 1 GPU smoke test |
| `CONFIG_PATH=launch/8gpu_dock.yaml bash launch/run.sh` | 8 GPU docking + ChEMBL prior + logP |
| `CONFIG_PATH=launch/8gpu_chembl_only.yaml bash launch/run.sh` | 8 GPU ChEMBL/prior run |
| `CONFIG_PATH=launch/7gpu_dock_classifier_v0_1.yaml bash launch/run.sh` | docking + classifier |
| `bash launch/run_diffsbdd_single_reward_matrix_20260612.sh` | DiffSBDD single-reward matrix |

Do not use the old `run_6gpu*.sh` / `run_8gpu*.sh` wrappers for new jobs unless
they are repaired first. They have been moved to
`launch/archive/legacy_diffsbdd_wrappers/`, overlap with `launch/run.sh`, and
still reference the obsolete helper now living at `scripts/yaml_launch_env.py`
(older records write `launch/yaml_launch_env.py`). Prefer:

```bash
CONFIG_PATH=launch/<config>.yaml bash launch/run.sh
```

### Legacy entrypoints

| Entrypoint | Status |
|---|---|
| `ddpo/main_diffsbdd_ddp.py` | Retained DiffSBDD DDP path |
| ~~`ddpo/main_diffsbdd_single_gpu.py`~~ | **REMOVED** — moved to `launch/archive/legacy_diffsbdd_wrappers/` |
| ~~`ddpo/main_diffsbdd.py`~~ | **REMOVED** — moved to `launch/archive/legacy_diffsbdd_wrappers/` |

## 5. DiffSBDD ESRPO (Endpoint Preference) And ESRPO-ChEMBL (Cross-source)

### ESRPO mainline

```bash
# train (full config; single-GPU example)
CUDA_VISIBLE_DEVICES=5 python ddpo/main_diffsbdd_global_dpo.py \
  --config launch/diffsbdd_global_esrpo_x_only_lr1e5_mainline.yaml --allow-small-pairs --epochs 1

# generate from a trained checkpoint
DPO_CHECKPOINT=<run>/checkpoints/diffsbdd-global-dpo-epoch=0001.pt GPU_ID=5 ATTEMPTS=1000 \
  bash launch/run_diffsbdd_global_esrpo_x_only_lr1e5_generation.sh
```

### ESRPO-ChEMBL (cross-source preference pairs)

Method/ledger: `docs/overview/DiffSBDD-ESRPO-ChEMBL版总审计.md`.

```bash
# 1) per-molecule R1-R5 rules for docked ChEMBL molecules
python analysis/crosssource_pairs_260929/compute_chembl_docked_rules.py --workers 18
# 2) ChEMBL dock poses -> DiffSBDD endpoint-store shards
python analysis/crosssource_pairs_260929/chembl_to_endpoint_shards.py --batch-size 1000
# 3) build pairs: SIZE-MATCHED is mandatory ('ring'/'none' only for ablations)
python analysis/crosssource_pairs_260929/build_crosssource_pairs.py \
  --loser-manifest-glob 'analysis/diffsbdd_structure_pairs_matched_260831/*pairs.csv' \
  --match-key size --min-gap 0.5 --output-dir analysis/crosssource_pairs_260929/pairs_size
# 4) train (ESRPO loss; trainer unchanged)
CUDA_VISIBLE_DEVICES=5 python ddpo/main_diffsbdd_global_dpo.py \
  --config launch/diffsbdd_esrpo_crosssource_size_smoke.yaml \
  --allow-small-pairs --epochs 1 --learning-rate 5e-5 --pair-count 3400
# 5) checkpoint sweep generation / 6) dock generated molecules (LF + no-H)
bash analysis/crosssource_pairs_260929/scan_size_ckpts.sh
python analysis/crosssource_pairs_260929/evaluate_gen_docking.py \
  --sdf OUT/crosssource_size_scan/step000800/molecules.sdf \
  --work-dir analysis/crosssource_pairs_260929/size_scan_dock/step000800
```

Notes:
- winners are ChEMBL unidock poses (RE-CmeB frame); losers are DiffSBDD endpoints from the
  existing 200k store - no trajectory is needed (DPO reads only `x / categorical / nodesxsample`).
- HA matching is mandatory: unrestricted pairing leaves a +11.4 heavy-atom gap and turns
  molecular size into an exploitable signal.
- evaluation set: loose sanitize / R1-R5 (ChEMBL HA10-30 target = 90.6%) / shape guards
  (0-ring rate, QED, single-fragment) / docking (clean & HA10-30, LF + no-H).

## 6. DiffSBDD Base Model Training And Generation

### Base model training

```bash
conda run -n dif python DiffSBDD/train.py --config <DiffSBDD/config.yaml>
```

### Ligand generation

```bash
conda run -n dif python DiffSBDD/generate_ligands.py <CHECKPOINT> \
  --pdbfile <PDB_FILE> \
  --outfile <OUTPUT_FILE> \
  --n_samples <N> \
  --batch_size <B>
```

### Inpaint / fixed-core generation

```bash
conda run -n dif python DiffSBDD/inpaint.py --help
```

Recent dual-R/fixed-core work is documented under:

- `analysis/dual_sidechain_honokiol_9pc_20260617/`
- `analysis/dual_sidechain_honokiol_9pc_20260618/`
- `work_records/26-6/260617/`
- `work_records/26-6/260618/260618-01 DiffGui双R侧链解耦推进分析.md`

## 7. ChEMBL MLP And D-MPNN Classifiers

### MLP classifier

```bash
conda run -n dif python chembl_classifier/data/extract_negatives.py
conda run -n dif python chembl_classifier/data/build_dataset.py
conda run -n dif python chembl_classifier/train.py
conda run -n dif python chembl_classifier/evaluate.py
```

Key configs and models:

- `chembl_classifier/config.yaml`
- `chembl_classifier/config_v0_3_1_scaffold_strongpoly.yaml`
- `chembl_classifier/models/chembl_classifier_selected_v0_3_1_scaffold_strongpoly.pt`

### D-MPNN classifier

```bash
conda run -n dif python chembl_classifier/dmpnn/train.py \
  --config chembl_classifier/config_dmpnn_v0_1_scaffold_strongpoly.yaml
```

Reward integration:

- `ddpo/rewards/chembl_dmpnn_reward.py`
- `test/test_chembl_dmpnn_data.py`
- `test/test_chembl_dmpnn_training.py`
- `test/test_chembl_dmpnn_reward.py`

## 8. Docking

UniDock is installed in the `dif` conda environment and is the current primary docking backend.

```bash
conda run -n dif python -m unidock_tools.main --help
```

Relevant code:

- `ddpo/rewards/docking.py`
- `ddpo/rewards/docking_preparation.py`
- `ddpo/rewards/docking_factory.py`
- `ddpo/rewards/docking_backends/`
- `scripts/docking/`

## 9. Tests And Smoke Checks

### Syntax check changed Python files

```bash
conda run -n dif python -m py_compile <file1.py> <file2.py>
```

### Focused tests

```bash
conda run -n dif pytest -q test/test_dataproto_behavior.py
conda run -n dif pytest -q test/test_diffgui_actor.py test/test_diffgui_replay.py
conda run -n dif pytest -q test/test_chembl_dmpnn_data.py test/test_chembl_dmpnn_reward.py
conda run -n dif pytest -q test/test_docking_preparation.py test/test_docking_reward_behavior.py
```

### Main DiffSBDD smoke

```bash
bash launch/smoke.sh
```

### DiffGui smoke

Use the DiffGui launch script and a non-conflicting GPU/port when needed:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
NPROC_PER_NODE=1 \
MASTER_PORT=<PORT> \
CONFIG_PATH=<launch/diffgui_config.yaml> \
OUTPUT_DIR=<OUT/run_dir> \
bash launch/run_diffgui_docking_train_ddp_smoke.sh
```

## 10. Analysis And Reporting

Analysis artifacts should live under `analysis/<topic>_<date>/` and include:

- `report.md`
- CSV/JSON source tables
- PNG/SVG plots embedded in the Markdown report
- SDF files when molecule-level evidence is needed

Task records should live under:

```text
work_records/<YYMM>/<YYMMDD>/<YYMMDD>-<seq> <title>.md
```

Daily summaries live under:

```text
work_days/
```

## 11. Current High-Value Analysis Directories

| Directory | Use |
|---|---|
| `analysis/diffgui_rl/diffgui_single_reward_analysis_20260616/` | DiffGui single-reward matrix analysis |
| `analysis/diffgui_rl/diffgui_rl_next_tests_20260618/` | DiffGui RL next-test plan |
| `analysis/diffgui_rl/diffgui_ring_last3_lr3e5_training_plots_20260611/` | DiffGui ring-last3 detailed plots |
| `analysis/diffsbdd_rl/diffsbdd_single_reward_matrix_20260613/` | DiffSBDD single-reward matrix |
| `analysis/classifier/` | Classifier training/curation/figures |
| `analysis/benchmark/` | Cross-model benchmark |
| `analysis/dual_sidechain_honokiol_9pc_20260617/` | DiffSBDD dual-R fixed-core attempts |
| `analysis/dual_sidechain_honokiol_9pc_20260618/` | DiffGui dual-R feasibility analysis |

## 12. Operational Notes

- Active launch configs are under `launch/`, not top-level `configs/`.
- Formal outputs go to `runs/`.
- `analysis/` is for interpreted artifacts and should embed charts in reports.
- `output/` is not an active root and should not be used for new run outputs.
- For live progress checks, trust logs, rank epoch artifacts, checkpoints, and GPU process evidence before stale manifests.
