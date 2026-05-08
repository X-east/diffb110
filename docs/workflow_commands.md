# Workflow Commands

**最后更新**: 2026-05-08

This document records the current verified workflow entrypoints, launch scripts, and supporting utilities.

## 1. Environment

All project commands should be run in the `dif` environment.

~~~bash
conda activate dif
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
~~~

## 2. Primary Runtime Method

### Current verified 8-GPU docking RL launcher (production)
~~~bash
bash scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh
~~~

This launcher drives:
- `torchrun --nproc_per_node=8 ddpo/main_diffsbdd_ddp.py`
- config `configs/qed_sa_docking_8gjl_8gpu.yaml`
- docking backend `unidock`
- target protein 8GJL

### Current verified single-GPU debugging entrypoint
~~~bash
python ddpo/main_diffsbdd_single_gpu.py --config_path configs/qed_sa_1gpu.yaml
~~~

### Other launch scripts
~~~bash
bash scripts/run_rl_qed_sa_1gpu.sh       # 1-GPU, no docking
bash scripts/run_rl_qed_sa_2gpu.sh       # 2-GPU, no docking
bash scripts/run_rl_qed_sa_4gpu.sh       # 4-GPU, no docking
bash scripts/run_rl_qed_sa_8gpu.sh       # 8-GPU, no docking
~~~

### Secondary / legacy RL training entrypoint
~~~bash
python ddpo/main_diffsbdd.py --config_path <CONFIG_FILE>
~~~

Notes:
- `ddpo/main_diffsbdd.py` contains Ray-related logic (legacy)
- Lightning-based entrypoints do not exist in current tree

## 3. Verified Python Entrypoints

### Primary 8-GPU DDP training entrypoint
- `ddpo/main_diffsbdd_ddp.py`

CLI:
~~~bash
torchrun --nproc_per_node=8 ddpo/main_diffsbdd_ddp.py --config_path configs/qed_sa_docking_8gjl_8gpu.yaml
~~~

### Single-GPU debugging entrypoint
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
python ddpo/main_diffsbdd.py --config_path <CONFIG_FILE>
~~~

### DiffSBDD base model training
~~~bash
python DiffSBDD/train.py --config <CONFIG_YAML>
~~~

### DiffSBDD ligand generation
~~~bash
python DiffSBDD/generate_ligands.py <CHECKPOINT> \
    --pdbfile <PDB_FILE> \
    --outfile <OUTPUT_FILE> \
    --n_samples <N>
~~~

## 4. Current YAML Configurations

| Config | GPUs | Docking | Backend | Description |
|--------|------|---------|---------|-------------|
| `configs/qed_sa_docking_8gjl_8gpu.yaml` | 8 | Yes | UniDock | ★ Current production config |
| `configs/qed_sa_1gpu.yaml` | 1 | No | - | Single-GPU debug |
| `configs/qed_sa_2gpu.yaml` | 2 | No | - | 2-GPU |
| `configs/qed_sa_2gpu_smoke.yaml` | 2 | No | - | 2-GPU smoke test |
| `configs/qed_sa_4gpu_16mol_smoke.yaml` | 4 | No | - | 4-GPU smoke test |
| `configs/qed_sa_8gpu_16mol_50ep.yaml` | 8 | No | - | 8-GPU no docking |

## 5. Entry-Point Relationship

In the current repository:

- **Production 8-GPU RL docking**: `scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh` → `ddpo/main_diffsbdd_ddp.py`
- **Single-GPU debugging**: `ddpo/main_diffsbdd_single_gpu.py`
- **Legacy**: `ddpo/main_diffsbdd.py` (contains Ray logic)
- **Lightning**: not present in current tree — do not reference as active path

### Underlying key Python modules
- `ddpo/main_diffsbdd_ddp.py` — DDP training entry
- `ddpo/main_diffsbdd_single_gpu.py` — single-GPU entry
- `ddpo/integration/diffsbdd_model.py` — DiffSBDDModel adapter
- `ddpo/runtime/build.py` — component factory
- `ddpo/runtime/rollout.py` — trajectory sampling
- `RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py` — EDMActor (PPO update)
- `ddpo/kl/gaussian_kl.py` — KL divergence computation
- `ddpo/rewards/` — reward system

## 6. Repository-Internal Launch Scripts

~~~bash
bash scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh   # ★ Current production
bash scripts/run_rl_qed_sa_1gpu.sh
bash scripts/run_rl_qed_sa_2gpu.sh
bash scripts/run_rl_qed_sa_4gpu.sh
bash scripts/run_rl_qed_sa_8gpu.sh
bash scripts/run_unidock_box_8GJL.sh               # UniDock box setup
~~~

Legacy scripts (may contain Ray logic):
~~~bash
bash scripts/test_rl_qed_sa.py                       # Test utility
~~~

## 7. Current Operational Status

- Verified project environment: `dif`
- Runtime project root: `/home/md666/data/disk6/DiffB110`
- Current production workflow: `torchrun` + `ddpo/main_diffsbdd_ddp.py` + UniDock
- Lightning is **not** the current training path
- Ray is **deprecated** — no new Ray dependencies

## 8. Known Issues

### PyTorch version compatibility
- Current: `torch==2.0.1+cu118`
- Some components expect `PyTorch >= 2.2` — produces warnings but does not prevent execution

### Legacy Ray assumptions
- `ddpo/main_diffsbdd.py` still contains Ray logic
- Legacy scripts in `something/old-scripts/` reference Ray

### KL divergence known gaps
- `compute_gaussian_kl()` lacks NaN/Inf protection — extreme σ can cause training explosion
- No separate KL logging — cannot monitor KL contribution independently
- Both issues should be fixed before enabling `kl_coef > 0`
