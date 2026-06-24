# Project Paths

> **最后更新**: 2026-06-18

This file records the current directory layout and canonical path conventions for the DiffB110 workspace.

## Canonical Roots

| Name | Path | Notes |
|---|---|---|
| `project_root` | `/data/disk6/DiffB110` | Canonical documentation path |
| `runtime_project_root` | `/home/md666/data/disk6/DiffB110` | Runtime-visible path used in this shell |
| `repo_root` | `/home/md666/data/disk6/DiffB110` | Current checkout |

Use runtime-visible absolute paths in local commands and work records when clarity matters.

## Core Code

| Name | Path | Notes |
|---|---|---|
| `ddpo_root` | `/home/md666/data/disk6/DiffB110/ddpo` | DDPO/PPO runtime, rewards, integrations |
| `ddpo_rewards_root` | `/home/md666/data/disk6/DiffB110/ddpo/rewards` | Reward implementations |
| `ddpo_diffgui_root` | `/home/md666/data/disk6/DiffB110/ddpo/diffgui` | Current DiffGui RL runtime |
| `ddpo_runtime_root` | `/home/md666/data/disk6/DiffB110/ddpo/runtime` | Retained DiffSBDD runtime |
| `diffsbdd_root` | `/home/md666/data/disk6/DiffB110/DiffSBDD` | DiffSBDD model code and retained baseline |
| `chembl_classifier_root` | `/home/md666/data/disk6/DiffB110/chembl_classifier` | MLP and D-MPNN classifiers |

## Launch And Runtime Outputs

| Name | Path | Notes |
|---|---|---|
| `launch_root` | `/home/md666/data/disk6/DiffB110/launch` | Active YAML and shell launch entrypoints |
| `launch_archive_root` | `/home/md666/data/disk6/DiffB110/launch/archive` | Non-active launch scripts, historical configs, generated matrices, logs, caches |
| `launch_generated_root` | `/home/md666/data/disk6/DiffB110/launch/archive/generated` | Regenerable experiment matrices, not hand-maintained entrypoints |
| `launch_logs_root` | `/home/md666/data/disk6/DiffB110/launch/archive/logs` | Batch launcher logs; runtime evidence, not launch APIs |
| `smoke_test` | `/home/md666/data/disk6/DiffB110/launch/smoke.sh` | Standard DiffSBDD smoke |
| `outputs_root` | `/home/md666/data/disk6/DiffB110/OUT` | Formal training/generation/smoke output root |

`output/` is not an active project output root. Historical files should be moved into `work_days/`, `work_records/`, or the relevant `analysis/` directory.

## Data And Models

| Name | Path | Notes |
|---|---|---|
| `workspace_data_root` | `/home/md666/data/disk6/DiffB110/data` | Protein, ligand, ChEMBL, CrossDocked, RND-EPI data |
| `chembl_data_root` | `/home/md666/data/disk6/DiffB110/data/chembl` | ChEMBL baseline data |
| `cache_root` | `/home/md666/data/disk6/DiffB110/cache` | Cache files |
| `top_level_checkpoints_root` | `/home/md666/data/disk6/DiffB110/checkpoints` | Small top-level checkpoint area |
| `diffsbdd_checkpoints_root` | `/home/md666/data/disk6/DiffB110/DiffSBDD/checkpoints` | DiffSBDD checkpoints |
| `chembl_classifier_models` | `/home/md666/data/disk6/DiffB110/chembl_classifier/models` | MLP/D-MPNN model files and datasets |

Common ChEMBL files:

- `/home/md666/data/disk6/DiffB110/data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz`
- `/home/md666/data/disk6/DiffB110/data/chembl/chembl_36_fcfp4_fingerprints.pkl.gz`
- `/home/md666/data/disk6/DiffB110/data/chembl/chembl_36_properties.csv.gz`

## Analysis And Records

| Name | Path | Notes |
|---|---|---|
| `analysis_root` | `/home/md666/data/disk6/DiffB110/analysis` | Analysis artifacts, reports, plots, CSV/JSON/SDF |
| `diffgui_rl_analysis_root` | `/home/md666/data/disk6/DiffB110/analysis/diffgui_rl` | Current DiffGui RL analysis artifacts |
| `diffgui_rl_major_summary` | `/home/md666/data/disk6/DiffB110/analysis/diffgui_rl_major_runs_20260618` | Current DiffGui RL major-run summary pipeline |
| `work_records_root` | `/home/md666/data/disk6/DiffB110/work_records` | Task-level records |
| `work_days_root` | `/home/md666/data/disk6/DiffB110/work_days` | Daily/weekly/monthly summaries |
| `docs_root` | `/home/md666/data/disk6/DiffB110/docs` | Current docs |
| `materials_root` | `/home/md666/data/disk6/DiffB110/materials` | Historical design notes archive |

Analysis reports should embed plots and tables directly. CSV/JSON/SDF files are supporting evidence, not replacements for in-report summaries.

## Reference Projects

| Name | Path | Notes |
|---|---|---|
| `reference_projects_root` | `/home/md666/data/disk6/DiffB110/reference project` | External method references |
| `diffgui_reference_root` | `/home/md666/data/disk6/DiffB110/reference project/DiffGui` | DiffGui reference model and scripts |
| `deppa_reference_root` | `/home/md666/data/disk6/DiffB110/reference project/DePPA` | DePPA reference |
| `molskill_reference_root` | `/home/md666/data/disk6/DiffB110/reference project/molskill` | MolSkill reference |

Reference projects are method references and environment-specific. Do not install their dependencies into the main `dif` environment without a deliberate compatibility plan.

## Environment Names

| Environment | Use |
|---|---|
| `dif` | Main DiffB110 / DiffSBDD / reward / analysis environment |
| `diffgui` | DiffGui reference model loading, DiffGui generation, DiffGui PPO/DDPO |

DiffGui reward-only SDF scoring may use the main environment only when it does not import the reference DiffGui model. Any DiffGui model path must use the DiffGui environment.

## Tests

| Name | Path |
|---|---|
| `tests_root` | `/home/md666/data/disk6/DiffB110/test` |

## Notes

- Active training configs are under `launch/`, not top-level `configs/`.
- For new DiffGui RL training, prefer the `launch/run_diffgui_*.sh` / `launch/start_diffgui_*.sh` wrappers that call `ddpo/main_diffgui_docking_ddp.py` in the `diffgui` environment.
- For retained DiffSBDD training, prefer `CONFIG_PATH=launch/<config>.yaml bash launch/run.sh` over older 6/8GPU wrapper scripts.
- Formal run evidence is under `OUT/`.
- Research conclusions usually require `analysis/` plus `work_records/`.
- Top-level `output/` should not be reintroduced as a live output root.
