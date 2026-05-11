# Project Paths

## Canonical Root
- `project_root=/data/disk6/DiffB110`
- `repo_root=/data/disk6/DiffB110`
- `docs_root=/data/disk6/DiffB110/docs`
- `scripts_root=/data/disk6/DiffB110/scripts`
- `cursor_rules_root=/data/disk6/DiffB110/.cursor/rules`

## Runtime-Visible Root
- `runtime_project_root=/home/md666/data/disk6/DiffB110`

## User-Maintained Launch Scripts
- `run_6gpu_sh=/home/md666/data/disk6/DiffB110/run_6gpu.sh`
- `run_4gpu_test_sh=/home/md666/data/disk6/DiffB110/run_4gpu_test.sh`
- `run_4gpu_druglikeness_only_sh=/home/md666/data/disk6/DiffB110/run_4gpu_druglikeness_only.sh`

## Main Project Modules
- `ddpo_root=/data/disk6/DiffB110/ddpo`
- `ddpo_configs_root=/data/disk6/DiffB110/ddpo/configs`
- `ddpo_rewards_root=/data/disk6/DiffB110/ddpo/rewards`
- `ddpo_trainer_root=/data/disk6/DiffB110/ddpo/trainer`
- `ddpo_utils_root=/data/disk6/DiffB110/ddpo/utils`

- `diffsbdd_root=/data/disk6/DiffB110/DiffSBDD`
- `diffsbdd_configs_root=/data/disk6/DiffB110/DiffSBDD/configs`
- `diffsbdd_checkpoints_root=/data/disk6/DiffB110/DiffSBDD/checkpoints`
- `diffsbdd_analysis_root=/data/disk6/DiffB110/DiffSBDD/analysis`

- `rlpf_root=/data/disk6/DiffB110/RLPF-verl_diffusion`
- `rlpf_exp_root=/data/disk6/DiffB110/RLPF-verl_diffusion/exp`
- `rlpf_model_root=/data/disk6/DiffB110/RLPF-verl_diffusion/Model`

- `unidock_repo_root=/data/disk6/DiffB110/Uni-Dock`
- `unidock_tools_root=/data/disk6/DiffB110/Uni-Dock/unidock_tools`
- `unidock_src_root=/data/disk6/DiffB110/Uni-Dock/unidock/src`

- `molformer_root=/data/disk6/DiffB110/molformer-main`

## Data and Cache
- `workspace_data_root=/data/disk6/DiffB110/data`
- `cache_root=/data/disk6/DiffB110/cache`

## Outputs and Logs
- `outputs_root=/data/disk6/DiffB110/OUT`
- `outputs_log_root=/data/disk6/DiffB110/OUT/log`
- `outputs_out_root=/data/disk6/DiffB110/OUT/out`

## Tests
- `tests_root=/data/disk6/DiffB110/tests`
- `tests_unit_root=/data/disk6/DiffB110/tests/unit`
- `tests_integration_root=/data/disk6/DiffB110/tests/integration`
- `tests_property_root=/data/disk6/DiffB110/tests/property`

## Notes
- Use canonical paths under `/data/disk6/DiffB110` for documentation and code changes.
- The user’s daily runtime launch scripts are currently called via `/home/md666/data/disk6/DiffB110/...`.
- Do not mix `/home/md666/data/disk6/DiffB110` with `/data/disk6/DiffB110` inside the same script unless the path mapping is intentional and verified.
- The repository contains multiple subprojects: `ddpo`, `DiffSBDD`, `RLPF-verl_diffusion`, `Uni-Dock`, and `molformer-main`.
- The preferred user-facing runtime entrypoints are:
  - `run_6gpu.sh`
  - `run_4gpu_test.sh`
  - `run_4gpu_druglikeness_only.sh`
- Paths such as `protein_root`, `ligand_root`, `pocket_root`, `reference_ligand_root`, `checkpoint_root`, and `docking_root` should be added only after confirming actual folder locations.