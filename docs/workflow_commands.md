# Workflow Commands

> **最后更新**: 2026-06-18

This document records the current verified workflow entrypoints, launch scripts, and supporting utilities in the repository.

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

Use `diffgui` for any task that imports the reference DiffGui model, loads `reference project/DiffGui/ckpt/trained.pt`, performs DiffGui generation, or runs DiffGui PPO/DDPO.

Do not install DiffGui-only dependencies into `dif`.

## 2. DiffGui PPO/DDPO Current Main Line

DiffGui has its own decoupled runtime under `ddpo/diffgui/` and must run in the `diffgui` environment when the reference model is loaded. This is the current reinforcement-learning main line.

### Main Python entrypoints

```bash
torchrun --nproc_per_node=<N> ddpo/main_diffgui_docking_ddp.py --config <launch/config.yaml>
python ddpo/main_diffgui_reward_only.py --config <launch/config.yaml>
```

### Launch scripts

| Command | Use |
|---|---|
| `bash launch/run_diffgui_docking_train_ddp_smoke.sh` | DiffGui DDP smoke |
| `bash launch/run_diffgui_deppa_last3_lr1e5_b18x8_t200_100ep_20260623.sh` | Current DePPA-style DiffGui RL formal run |
| `bash launch/run_diffgui_next6_matrix_20260618.sh` | DiffGui next6 serial matrix |
| `bash launch/run_diffgui_ring_control_matrix_20260615.sh` | QED/SA/ring/combination matrix |
| `bash launch/start_diffgui_next4_continue_to50_20260621.sh` | Continue next4 run to 50 total epochs |

Historical DiffGui launchers live under `launch/archive/diffgui_historical/`
and should be used only to reproduce earlier runs.

### Important DiffGui evidence paths

- `OUT/diffgui-*`
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

DiffGui RL stage-summary statistics are documented in:

- `docs/DiffGui强化学习统计清单.md`

## 3. DiffSBDD DDPO/PPO Retained Baseline

The retained DiffSBDD reinforcement-learning path uses PyTorch native DDP via `torchrun`. Use it for comparison, backup, or historical reproduction, not as the current research main line.

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
still reference the obsolete `launch/yaml_launch_env.py` helper in several
places. Prefer:

```bash
CONFIG_PATH=launch/<config>.yaml bash launch/run.sh
```

### Legacy entrypoints

| Entrypoint | Status |
|---|---|
| `ddpo/main_diffsbdd_ddp.py` | Retained DiffSBDD DDP path |
| `ddpo/main_diffsbdd_single_gpu.py` | Legacy single-GPU debug path |
| `ddpo/main_diffsbdd.py` | Legacy Ray-style reference |

## 4. DiffSBDD Base Model Training And Generation

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

## 5. ChEMBL MLP And D-MPNN Classifiers

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

## 6. Docking

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

## 7. Tests And Smoke Checks

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

## 8. Analysis And Reporting

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

## 9. Current High-Value Analysis Directories

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

## 10. Operational Notes

- Active launch configs are under `launch/`, not top-level `configs/`.
- Formal outputs go to `OUT/`.
- `analysis/` is for interpreted artifacts and should embed charts in reports.
- `output/` is not an active root and should not be used for new run outputs.
- For live progress checks, trust logs, rank epoch artifacts, checkpoints, and GPU process evidence before stale manifests.
