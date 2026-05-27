# Project Paths

> **最后更新**: 2026-05-27

## Canonical Root
- `project_root=/data/disk6/DiffB110`
- `repo_root=/data/disk6/DiffB110`
- `docs_root=/data/disk6/DiffB110/docs`
- `scripts_root=/data/disk6/DiffB110/scripts`
- `cursor_rules_root=/data/disk6/DiffB110/.cursor/rules`

## Runtime-Visible Root
- `runtime_project_root=/home/md666/data/disk6/DiffB110`

## Launch Configs (Active Training)
- `launch_root=/home/md666/data/disk6/DiffB110/launch`
- `launch_configs=/home/md666/data/disk6/DiffB110/launch/*.yaml`
- `launch_scripts=/home/md666/data/disk6/DiffB110/launch/*.sh`
- `smoke_test=/home/md666/data/disk6/DiffB110/launch/smoke.sh`

## Main Project Modules
- `ddpo_root=/data/disk6/DiffB110/ddpo`
- `ddpo_configs_root=/data/disk6/DiffB110/ddpo/configs`
- `ddpo_rewards_root=/data/disk6/DiffB110/ddpo/rewards`
- `ddpo_trainer_root=/data/disk6/DiffB110/ddpo/trainer`
- `ddpo_utils_root=/data/disk6/DiffB110/ddpo/utils`

- `chembl_classifier_root=/data/disk6/DiffB110/chembl_classifier`
- `chembl_classifier_data=/data/disk6/DiffB110/chembl_classifier/data`
- `chembl_classifier_models=/data/disk6/DiffB110/chembl_classifier/models`

- `diffsbdd_root=/data/disk6/DiffB110/DiffSBDD`
- `diffsbdd_configs_root=/data/disk6/DiffB110/DiffSBDD/configs`
- `diffsbdd_checkpoints_root=/data/disk6/DiffB110/DiffSBDD/checkpoints`
- `diffsbdd_analysis_root=/data/disk6/DiffB110/DiffSBDD/analysis`

- `analysis_root=/data/disk6/DiffB110/analysis`

- `rlpf_root=/data/disk6/DiffB110/reference project/RLPF-verl_diffusion`  *(历史参考，已移出 git 跟踪)*
- `rlpf_exp_root=/data/disk6/DiffB110/reference project/RLPF-verl_diffusion/exp`
- `rlpf_model_root=/data/disk6/DiffB110/reference project/RLPF-verl_diffusion/Model`

- `unidock_repo_root=/data/disk6/DiffB110/Uni-Dock`
- `unidock_tools_root=/data/disk6/DiffB110/Uni-Dock/unidock_tools`
- `unidock_src_root=/data/disk6/DiffB110/Uni-Dock/unidock/src`

- `molformer_root=/data/disk6/DiffB110/molformer-main`

## Data and Cache
- `workspace_data_root=/data/disk6/DiffB110/data`
- `cache_root=/data/disk6/DiffB110/cache`
- `chembl_data_root=/data/disk6/DiffB110/data/chembl`
- `chembl_fingerprint_ecfp4=/data/disk6/DiffB110/data/chembl/chembl_36_ecfp4.npz`
- `chembl_fingerprint_fcfp4=/data/disk6/DiffB110/data/chembl/chembl_36_fcfp4.npz`
- `chembl_properties=/data/disk6/DiffB110/data/chembl/chembl_36_properties.csv.gz`

## Outputs and Logs
- `outputs_root=/data/disk6/DiffB110/OUT`
- `outputs_log_root=/data/disk6/DiffB110/OUT/log`
- `outputs_out_root=/data/disk6/DiffB110/OUT/out`

## Tests
- `tests_root=/data/disk6/DiffB110/test`
- `tests_unit_root=/data/disk6/DiffB110/tests/unit`
- `tests_integration_root=/data/disk6/DiffB110/tests/integration`
- `tests_property_root=/data/disk6/DiffB110/tests/property`

## Notes
- Use canonical paths under `/data/disk6/DiffB110` for documentation and code changes.
- The user's daily runtime launch scripts are in `launch/*.sh` (torchrun-based DDP).
- The repository contains multiple subprojects: `ddpo`, `chembl_classifier`, `DiffSBDD`, `Uni-Dock`, and `molformer-main`.
- `RLPF-verl_diffusion` 已迁移至 `reference project/` 作为历史参考，不再由 git 跟踪。
- Launch configs (`launch/*.yaml`) are the primary configuration entrypoints, not `configs/`.