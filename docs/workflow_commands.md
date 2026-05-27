# Workflow Commands

> **最后更新**: 2026-05-27

This document records the current verified workflow entrypoints, launch scripts, and supporting utilities in the repository.

## 1. Environment

All project commands should be run in the `dif` environment.

~~~bash
conda activate dif
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
~~~

## 2. Primary Runtime Method (Multi-GPU DDP via torchrun)

The current verified training method uses **PyTorch native DDP** launched via `torchrun`.
All training configs live in `launch/`.

### Current training entrypoints

| Launch Script | GPUs | Reward Configuration |
|--------------|------|---------------------|
| `bash launch/6gpu_dock.sh` | 6 | docking + ChEMBL prior + logP distribution |
| `bash launch/6gpu_chembl_only.sh` | 6 | ChEMBL prior ECFP4+FCFP4 + logP + classifier |
| `bash launch/7gpu_classifier_only.sh` | 7 | Pure ChEMBL classifier only |
| `bash launch/8gpu_dock.sh` | 8 | docking + ChEMBL prior + logP distribution |
| `bash launch/8gpu_chembl_only.sh` | 8 | ChEMBL prior ECFP4+FCFP4 + logP |
| `bash launch/molskill_test.sh` | 8 | Pure MolSkill test |
| `bash launch/smoke.sh` | 1 | QED only, 1 epoch, auto-cleanup |

### Smoke test (mandatory after code changes)

~~~bash
bash launch/smoke.sh
~~~

Runs 1 epoch with batch=4 on 1 GPU, validates the full DDP pipeline, and auto-cleans GPU memory/port.

### Single GPU debugging (legacy)

~~~bash
python ddpo/main_diffsbdd_single_gpu.py --config_path configs/qed_sa_1gpu.yaml
~~~

## 3. Key Python Entrypoints

### RL / DDPO training entrypoint (current mainline)
- `ddpo/main_diffsbdd_ddp.py`

Launched via torchrun shell scripts in `launch/`. Direct invocation (rarely needed):
~~~bash
torchrun --nproc_per_node=6 --master_port=29500 ddpo/main_diffsbdd_ddp.py --config launch/6gpu_dock.yaml
~~~

### Single GPU entrypoint (legacy)
- `ddpo/main_diffsbdd_single_gpu.py`

CLI:
~~~bash
python ddpo/main_diffsbdd_single_gpu.py --config_path configs/qed_sa_1gpu.yaml
~~~

Arguments:
- `--config_path CONFIG_PATH` (required)
- `--save_path SAVE_PATH`
- `--seed SEED`
- `--log_level {DEBUG,INFO,WARNING,ERROR}`
- `--resume`
- `--checkpoint_path CHECKPOINT_PATH`

### Legacy Ray entrypoint
- `ddpo/main_diffsbdd.py`

CLI:
~~~bash
python ddpo/main_diffsbdd.py --help
~~~

Arguments: `--config_path`, `--save_path`, `--log_level`, `--seed`, `--resume`, `--checkpoint_path`

### ChEMBL classifier training pipeline
- `chembl_classifier/data/extract_negatives.py` — extract rule-based negative samples
- `chembl_classifier/data/build_dataset.py` — build featurized `dataset.pt`
- `chembl_classifier/train.py` — train the MLP classifier
- `chembl_classifier/evaluate.py` — evaluate AUROC / threshold analysis
- `chembl_classifier/config.yaml` — training hyperparameters

### DiffSBDD base model training
- `DiffSBDD/train.py`

CLI:
~~~bash
python DiffSBDD/train.py --help
~~~
Arguments: `--config CONFIG`, `--resume RESUME`

### DiffSBDD ligand generation
- `DiffSBDD/generate_ligands.py`

CLI:
~~~bash
python DiffSBDD/generate_ligands.py --help
~~~
Arguments: positional `checkpoint`, `--pdbfile`, `--resi_list`, `--ref_ligand`, `--outfile`, `--n_samples`, `--batch_size`, etc.

### Uni-Dock tool entrypoint
- `Uni-Dock/unidock_tools/src/unidock_tools/main.py`

CLI:
~~~bash
python Uni-Dock/unidock_tools/src/unidock_tools/main.py --help
~~~
Subcommands: `proteinprep`, `unidock_pipeline`, `mcdock`, `ligandprep`

## 4. Entrypoint Relationship

In the current repository snapshot:

- The verified active RL training code path is `ddpo/main_diffsbdd_ddp.py` (PyTorch native DDP via `torchrun`)
- Launch is always via `bash launch/<config>.sh` (shell wrappers that set up torchrun with correct CUDA_VISIBLE_DEVICES, etc.)
- `ddpo/main_diffsbdd_single_gpu.py` is a legacy single-GPU entrypoint, useful for quick debugging
- `ddpo/main_diffsbdd.py` is the legacy Ray entrypoint, preserved for historical reference
- `ddpo/main_diffsbdd_lightning.py` — does not exist in this checkout, should be treated as historical

Underlying modules still used by the active DDP path:
- `ddpo/runtime/rollout.py` — rollout logic
- `ddpo/runtime/train_helpers.py` — training loop orchestration + molecular descriptor computation
- `ddpo/integration/diffsbdd_model.py` — DiffSBDD model adapter
- `ddpo/runtime/actor.py` — 当前 policy update（从 RLPF actor 本地化）

## 5. Repository-Internal or Legacy Launchers

The repository also contains additional launchers that may be useful for reference, debugging, or legacy workflows:

~~~bash
# Legacy shell scripts (some still contain Ray logic)
bash scripts/train_8gpu.sh
bash scripts/train_unidock_8gpu.sh
bash ddpo/configs/run_training.sh
bash ddpo/configs/run_training_ddp.sh
~~~

Notes:
- These scripts remain in the repository for reference or legacy use.
- Some of them still contain Ray-related logic.
- They should not be treated as the primary production workflow.**

## 6. DiffSBDD Commands

### Base model training
~~~bash
python DiffSBDD/train.py --config <CONFIG_YAML>
~~~

### Ligand generation
~~~bash
python DiffSBDD/generate_ligands.py <CHECKPOINT> \
    --pdbfile <PDB_FILE> \
    --outfile <OUTPUT_FILE> \
    --n_samples <N>
~~~

## 7. Testing Commands

### Smoke test (mandatory after code changes)
~~~bash
bash launch/smoke.sh
~~~
1 GPU / 1 epoch / batch=4 / QED only. Validates full DDP pipeline without crashing. Auto-cleans GPU memory.

### Unit / integration tests
~~~bash
python test/test_le_reward_integration.py
python test/test_reward_consistency.py
~~~

## 8. ChEMBL Classifier Commands

~~~bash
# Extract negative samples from ChEMBL (property rule-based)
cd chembl_classifier
python data/extract_negatives.py

# Build featurized dataset (2220D Morgan FP + RDKit descriptors)
python data/build_dataset.py

# Train the MLP classifier
python train.py

# Evaluate on test set
python evaluate.py
~~~

Training hyperparameters in `chembl_classifier/config.yaml`:
- `model`: 2220 → 512 → 128 → 1 (BatchNorm + ReLU + Dropout)
- `batch_size`: 256
- `epochs`: 100
- `train_split`: 0.8

## 9. Analysis Commands

~~~bash
# Analyze training output
python analysis/analyze_8gpu_chembl_only.py
python analysis/analyze_7gpu_dock_ecfp4_fcfp4.py
python analysis/analyze_7gpu_dock_ecfp4_fcfp4_20260526.py
python analysis/analyze_reward_influence.py
python analysis/analyze_fingerprint_exps.py

# Historical EDM evaluation from the reference RLPF tree
python RLPF-verl_diffusion/Model/EDM/eval_analyze.py
~~~

## 10. Current Operational Status

- Verified project environment: `dif`
- Canonical project root: `/home/md666/data/disk6/DiffB110`
- Active training entrypoint: `ddpo/main_diffsbdd_ddp.py` (PyTorch native DDP)
- Launch mechanism: `bash launch/<config>.sh` (torchrun-based)
- Primary docking backend: UniDock
- Core reward paradigm: high-dimensional feature classifier (2220D MLP) + fingerprint priors + distribution matching
- Monitoring: 22 molecular descriptors per epoch
- The repository still contains Ray-based logic in several legacy scripts.
- References to Lightning DDP in older docs are now historical.

## 11. Known Entry-Point and Environment Issues

### PyTorch version compatibility warning
Current environment provides `torch==2.0.1+cu118`. Some imported components expect `PyTorch >= 2.2`. This does not prevent execution but is an environment compatibility risk.

### Uni-Dock local package conflict
Running `python Uni-Dock/unidock_tools/src/unidock_tools/application/unidock_pipeline.py --help` fails because Python imports the installed `unidock_tools` package from the environment instead of the local repository source.

### Legacy Ray assumptions
The following scripts still assume Ray-based orchestration:
- `scripts/train_8gpu.sh`
- `scripts/train_unidock_8gpu.sh`
- `ddpo/configs/run_training.sh`
- `ddpo/configs/run_training_ddp.sh`

### Legacy script argument mismatch
`scripts/train_unidock_8gpu.sh` calls `--config` but `ddpo/main_diffsbdd.py` expects `--config_path`.
