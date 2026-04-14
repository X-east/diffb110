# Workflow Commands

This document records the current verified workflow entrypoints, launch scripts, and supporting utilities in the repository.

## 1. Environment

All project commands should be run in the `dif` environment.

~~~bash
conda activate dif
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
~~~

## 2. Primary Runtime Method

The current verified project workflow in this repository snapshot is centered on direct Python execution of the custom single-process training entrypoint.

### Current verified training entrypoint
~~~bash
python ddpo/main_diffsbdd_single_gpu.py --config_path configs/qed_sa_1gpu.yaml
~~~

### Secondary / legacy RL training entrypoint
~~~bash
python ddpo/main_diffsbdd.py --config_path <CONFIG_FILE>
~~~

Notes:
- older docs referenced bash launchers such as `run_6gpu.sh`, `run_4gpu_test.sh`, and `run_4gpu_druglikeness_only.sh`
- these scripts were not found in the current repository root during verification
- treat those launcher references as historical / external-worktree information unless the scripts reappear or their location is confirmed

## 3. Verified Python Entrypoints

These Python entrypoints are currently the most important paths for inspection, debugging, and day-to-day execution in this repository snapshot.

### Preferred current RL / DDPO training entrypoint
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

### Secondary / legacy RL training entrypoint
- `ddpo/main_diffsbdd.py`

CLI:
~~~bash
python ddpo/main_diffsbdd.py --help
~~~

Arguments:
- `--config_path CONFIG_PATH`
- `--save_path SAVE_PATH`
- `--log_level {DEBUG,INFO,WARNING,ERROR}`
- `--seed SEED`
- `--resume`
- `--checkpoint_path CHECKPOINT_PATH`

### Historical / documentation-only reference
- `ddpo/main_diffsbdd_lightning.py`

Status:
- referenced by older documents as a Lightning DDP entrypoint
- not found in the current repository tree
- should not be treated as a verified active entrypoint in this checkout

### DiffSBDD base model training
- `DiffSBDD/train.py`

CLI:
~~~bash
python DiffSBDD/train.py --help
~~~

Arguments:
- `--config CONFIG`
- `--resume RESUME`

### DiffSBDD ligand generation
- `DiffSBDD/generate_ligands.py`

CLI:
~~~bash
python DiffSBDD/generate_ligands.py --help
~~~

Arguments:
- positional: `checkpoint`
- `--pdbfile PDBFILE`
- `--resi_list RESI_LIST [RESI_LIST ...]`
- `--ref_ligand REF_LIGAND`
- `--outfile OUTFILE`
- `--n_samples N_SAMPLES`
- `--batch_size BATCH_SIZE`
- `--num_nodes_lig NUM_NODES_LIG`
- `--all_frags`
- `--sanitize`
- `--relax`
- `--resamplings RESAMPLINGS`
- `--jump_length JUMP_LENGTH`
- `--timesteps TIMESTEPS`

### Uni-Dock tool entrypoint
- `Uni-Dock/unidock_tools/src/unidock_tools/main.py`

CLI:
~~~bash
python Uni-Dock/unidock_tools/src/unidock_tools/main.py --help
~~~

Subcommands:
- `proteinprep`
- `unidock_pipeline`
- `mcdock`
- `ligandprep`

## 4. Entrypoint Relationship

In the current repository snapshot:

- the verified active RL training code path is the custom single-process entrypoint `ddpo/main_diffsbdd_single_gpu.py`
- `ddpo/main_diffsbdd.py` remains a secondary / legacy RL path
- older documentation references to `ddpo/main_diffsbdd_lightning.py` should be treated as historical or from another checkout unless that file is restored

Underlying Python entrypoints still exist and are useful for inspection or debugging.

### Underlying RL / DDPO entrypoints
- `ddpo/main_diffsbdd_single_gpu.py`
- `ddpo/main_diffsbdd.py`
- `RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py` (policy update path used by the active single-process flow)

### Underlying DiffSBDD entrypoints
- `DiffSBDD/train.py`
- `DiffSBDD/generate_ligands.py`

### Underlying Uni-Dock entrypoint
- `Uni-Dock/unidock_tools/src/unidock_tools/main.py`

## 5. Repository-Internal or Legacy Launchers

The repository also contains additional launchers that may be useful for reference, debugging, or legacy workflows:

~~~bash
bash scripts/train_8gpu.sh
bash scripts/train_unidock_8gpu.sh
bash ddpo/configs/run_training.sh
bash ddpo/configs/run_training_ddp.sh
~~~

Notes:
- These scripts remain in the repository for reference or legacy use.
- Some of them still contain Ray-related logic.
- They should not be treated as the primary production workflow unless explicitly needed.

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

## 7. Docking / Reward / Analysis Utilities

### Visualization / generation helper
~~~bash
python scripts/generate_and_visualize.py --help
~~~

### Reward and docking test utilities
~~~bash
python scripts/test_docking_reward.py
python scripts/test_unidock_docking.py
python scripts/test_unidock_reward.py
python scripts/test_druglikeness_reward.py
~~~

### Analysis utilities
~~~bash
python scripts/analyze_generated_data.py
python RLPF-verl_diffusion/Model/EDM/eval_analyze.py
python DiffSBDD/analysis/docking.py
~~~

## 8. Validation / Debug Utilities

~~~bash
python scripts/verify_integration.py
python scripts/verify_model_interface.py
python scripts/verify_unidock.py
python scripts/verify_task5_checkpoint.py
python scripts/complete_model_verification.py
python scripts/test_rl_loop.py
python scripts/test_rl_pipeline.py
python scripts/test_diffsbdd_model_e2e.py
python scripts/test_unidock_integration.py
~~~

## 9. Current Operational Status

- Verified project environment: `dif`
- Canonical project root: `/data/disk6/DiffB110`
- Common shell-visible runtime path currently used by the user: `/home/md666/data/disk6/DiffB110`
- The repository still contains Ray-based logic in several legacy scripts and launchers.
- The current intended scientific direction remains GPU-based docking/scoring without introducing new Ray dependencies.
- The current verified training mainline in this checkout is `ddpo/main_diffsbdd_single_gpu.py` plus:
  - `ddpo/runtime/rollout.py`
  - `ddpo/integration/diffsbdd_model.py`
  - `RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py`
- References to Lightning DDP in older docs should currently be interpreted as historical / planned rather than verified-present code in this tree.

## 10. Known Entry-Point and Environment Issues

### PyTorch version compatibility warning
Several Python entrypoints print a warning indicating that one imported component expects `PyTorch >= 2.2`, while the current environment provides:

- `torch==2.0.1+cu118`

This does not prevent `--help` from running, but it should be recorded as an environment compatibility risk.

### Uni-Dock local package conflict
Running:

~~~bash
python Uni-Dock/unidock_tools/src/unidock_tools/application/unidock_pipeline.py --help
~~~

currently fails because Python imports the installed `unidock_tools` package from the environment instead of the local repository source.

### Legacy Ray assumptions
The following scripts still assume Ray-based orchestration or still contain Ray-related checks:

- `scripts/train_8gpu.sh`
- `scripts/train_unidock_8gpu.sh`
- `ddpo/configs/run_training.sh`
- `ddpo/configs/run_training_ddp.sh`

### Legacy script argument mismatch
`scripts/train_unidock_8gpu.sh` calls:

~~~bash
python ddpo/main_diffsbdd.py --config ddpo/configs/unidock_8gpu_training.yaml
~~~

but the verified CLI of `ddpo/main_diffsbdd.py` expects:

~~~bash
python ddpo/main_diffsbdd.py --config_path <CONFIG_FILE>
~~~

Therefore, that script should be treated as needing correction before production use.

## 11. Runtime Notes

- For the current repository snapshot, prefer the verified Python entrypoints documented above for daily execution, inspection, and debugging.
- Do not assume the previously referenced bash launch scripts are available unless their paths are re-confirmed in the active working tree.
- For code understanding, debugging, and Cursor assistance, the Python entrypoints listed above remain the authoritative internal interfaces.
- When documenting or editing workflows, distinguish clearly between:
  - verified current Python entrypoints
  - historical / missing launch scripts referenced by older docs
  - legacy repository launchers