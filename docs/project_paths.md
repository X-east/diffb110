# Project Paths

**最后更新**: 2026-05-08

## Canonical Root
- `project_root=/data/disk6/DiffB110`
- `repo_root=/data/disk6/DiffB110`
- `docs_root=/data/disk6/DiffB110/docs`
- `scripts_root=/data/disk6/DiffB110/scripts`

## Runtime-Visible Root
- `runtime_project_root=/home/md666/data/disk6/DiffB110`

## User-Maintained Launch Scripts
- `run_8gpu_unidock_rl_sh=/home/md666/data/disk6/DiffB110/scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh`
- `run_1gpu_sh=/home/md666/data/disk6/DiffB110/scripts/run_rl_qed_sa_1gpu.sh`
- `run_2gpu_sh=/home/md666/data/disk6/DiffB110/scripts/run_rl_qed_sa_2gpu.sh`
- `run_4gpu_sh=/home/md666/data/disk6/DiffB110/scripts/run_rl_qed_sa_4gpu.sh`
- `run_8gpu_sh=/home/md666/data/disk6/DiffB110/scripts/run_rl_qed_sa_8gpu.sh`

## Main Project Modules
- `ddpo_root=/data/disk6/DiffB110/ddpo`
- `ddpo_configs_root=/data/disk6/DiffB110/ddpo/configs` (note: this is Python configs, not YAML)
- `ddpo_rewards_root=/data/disk6/DiffB110/ddpo/rewards`
- `ddpo_kl_root=/data/disk6/DiffB110/ddpo/kl`
- `ddpo_runtime_root=/data/disk6/DiffB110/ddpo/runtime`
- `ddpo_utils_root=/data/disk6/DiffB110/ddpo/utils`
- `ddpo_integration_root=/data/disk6/DiffB110/ddpo/integration`

- `diffsbdd_root=/data/disk6/DiffB110/DiffSBDD`
- `diffsbdd_configs_root=/data/disk6/DiffB110/DiffSBDD/configs`
- `diffsbdd_checkpoints_root=/data/disk6/DiffB110/DiffSBDD/checkpoints`
- `diffsbdd_analysis_root=/data/disk6/DiffB110/DiffSBDD/analysis`

- `rlpf_root=/data/disk6/DiffB110/RLPF-verl_diffusion`
- `rlpf_model_root=/data/disk6/DiffB110/RLPF-verl_diffusion/Model`

## YAML Configurations
- `configs_root=/data/disk6/DiffB110/configs`
- `config_production=/data/disk6/DiffB110/configs/qed_sa_docking_8gjl_8gpu.yaml`
- `config_1gpu=/data/disk6/DiffB110/configs/qed_sa_1gpu.yaml`
- `config_2gpu=/data/disk6/DiffB110/configs/qed_sa_2gpu.yaml`
- `config_8gpu_nodock=/data/disk6/DiffB110/configs/qed_sa_8gpu_16mol_50ep.yaml`

## Data and Cache
- `workspace_data_root=/data/disk6/DiffB110/data`
- `cache_root=/data/disk6/DiffB110/cache`

## Outputs
- `outputs_root=/data/disk6/DiffB110/OUT`

## Tests
- `tests_root=/data/disk6/DiffB110/test`
- Test files (flat directory):
  - `test/test_docking_preparation.py`
  - `test/test_docking_factory_policy.py`
  - `test/test_docking_reward_behavior.py`
  - `test/test_recording_mainline.py`
  - `test/test_add_progress_output.py`
  - `test/test_checkpoint.py`
  - `test/checkpoint_DeepSeek.py`

## Notes
- Use canonical paths under `/data/disk6/DiffB110` for documentation and code changes.
- The user's daily runtime uses `/home/md666/data/disk6/DiffB110/...`.
- Do not mix `/home/md666/data/disk6/DiffB110` with `/data/disk6/DiffB110` inside the same script.
- The repository contains subprojects: `ddpo`, `DiffSBDD`, `RLPF-verl_diffusion`.
- The preferred user-facing runtime entrypoint is `scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh`.
- Lightning-based paths (`main_diffsbdd_lightning.py`, `lightning_ddpo_module.py`) do not exist in the current tree.
- Legacy launchers (`run_6gpu.sh`, `run_4gpu_test.sh`, `run_4gpu_druglikeness_only.sh`) referenced in older docs do not exist in the current tree.
