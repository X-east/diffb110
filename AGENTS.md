# DiffB110 Codex Project Guide

## Project Scope

This repository is the DiffB110 project: pocket-conditioned 3D molecule
generation for the RE-CmeB / CmeB target, with the current research main line
focused on DiffGui DDPO/PPO reinforcement learning. Treat
`ddpo/main_diffgui_docking_ddp.py` as the current DiffGui RL training entry
point. `ddpo/main_diffsbdd_ddp.py` remains the retained DiffSBDD RL baseline and
comparison path.

Before changing project code, read:

- `docs/project_context.md`
- `docs/workflow_commands.md`
- the relevant module near the requested change

## Evidence-Based Change Rule

- All model, sampler, reconstruction, geometry, atom-prior, edge-diffusion, and
  reinforcement-learning change directions must be grounded in an existing
  program reference before implementation.
- Acceptable references are: existing DiffB110/DiffSBDD code, local reference
  projects under `reference project/`, downloaded third-party source code, or a
  public paper/repository/documentation page.
- Before implementing a nontrivial scientific change, record the reference
  source, what is being borrowed, what is different in this project, expected
  benefit, risk, and the validation metric or ablation that will decide whether
  the change stays.
- Do not introduce model-internal scientific mechanisms only from intuition. If
  no suitable reference exists, first create an isolated exploratory prototype
  and report it as speculative rather than merging it into the main pipeline.

## Worktree Safety

- Run `git status --short` before code changes.
- If uncommitted changes exist, report the relevant status to the user before
  editing and do not overwrite, revert, or reformat unrelated changes.
- Keep edits scoped to the requested task.
- Do not use destructive git commands unless the user explicitly asks.

## Environment

- Run project Python, tests, and training commands in the `dif` environment.
- Prefer `conda run -n dif <command>` for one-off commands when activation state
  is uncertain.
- In an independent git worktree, every DiffSBDD offline-IQL Python, pytest,
  preprocessing, and training command must prepend the active worktree root to
  `PYTHONPATH`, for example
  `PYTHONPATH="$(git rev-parse --show-toplevel):${PYTHONPATH:-}" conda run -n dif ...`.
  Do not rely on `conda run` to select the current worktree package: it can
  otherwise import `ddpo` from another checkout. Offline-IQL launch scripts must
  export this value themselves before invoking Python or `torchrun`.
- DiffGui reinforcement-learning work is environment-isolated from DiffSBDD.
  Any task that loads the reference DiffGui model/checkpoint, runs DiffGui
  denovo/fragment sampling, or trains DiffGui PPO/DDPO must use a dedicated
  DiffGui environment instead of `dif`. Do not install DiffGui-only dependencies
  into `dif`, and do not mix DiffGui launch scripts, checkpoints, or run outputs
  with the DiffSBDD RL runtime.
- DiffGui reward-only SDF scoring may use the regular project environment only
  when it does not import the reference DiffGui model. If a task touches the
  DiffGui model code path, run the DiffGui dependency preflight first and record
  the environment name in the run log.
- On this machine, do not use `rg` / `rg --files`: the installed ripgrep is
  launched through snap and may fail with `snap-confine` capability errors.
  Use `grep`, `find`, `sed`, `ls`, or small targeted shell commands instead.
- For generation, benchmark, model inference, and training tasks, prefer GPU
  execution when CUDA is actually available in the selected environment and the
  task is expected to run faster on GPU. Fall back to CPU only when CUDA is not
  visible, GPU memory is insufficient, or the task is CPU-bound; record the
  fallback reason in reports.
- For molecule generation specifically, do not treat sandboxed
  `torch.cuda.is_available() == False` as proof that GPU generation is
  impossible when `nvidia-smi` shows usable GPUs. In that case, request
  escalated/non-sandbox execution and run generation on GPU; CPU molecule
  generation is only acceptable after a real GPU attempt fails or the user
  explicitly asks for CPU. CPU-only generation attempts must not be used for
  speed conclusions.
- For CPU-parallel analysis, preprocessing, descriptor calculation, dataset
  building, and similar project tasks, default to 18 CPU cores unless the user
  specifies another value or the tool has a safer project-specific limit.
- When a command supports it, prefer explicit worker/thread settings such as
  `--num-workers 18`, `n_jobs=18`, or common thread env vars
  `OMP_NUM_THREADS=18 MKL_NUM_THREADS=18 OPENBLAS_NUM_THREADS=18
  NUMEXPR_NUM_THREADS=18`.
- Use configured YAML paths and project helpers instead of hardcoded local paths.

## Analysis Artifacts

- For every project analysis task, generate plots by default in addition to
  CSV/JSON/Markdown outputs, unless the user explicitly asks for text-only
  analysis.
- Save plots into the same analysis directory as the report, preferably as
  `.png` for quick viewing and `.svg` when vector output is useful.
- Embed generated plots directly in the Markdown report with Markdown image
  syntax such as `![caption](plot.png)`. Do not list plot files only as plain
  links or paths.
- Include tabular results directly in the Markdown report as Markdown tables or
  concise summary tables. CSV/JSON/SDF paths are supporting evidence, not a
  substitute for in-report tables or figures.
- Reference raw data files from the Markdown report with relative links to files
  in the same analysis directory, so readers can open the exact CSV/JSON/SDF
  evidence after seeing the embedded figure/table in context.
- Plot filenames should be descriptive and stable, such as
  `global_distribution_comparison.png`, `run_drift_summary.png`, or
  `top_reward_failure_modes.png`.
- When a task also needs a durable work record, use the work record as the
  single narrative report by default. Keep `analysis/` for scripts, CSV/JSON,
  plots, SDFs, and other evidence files, and do not create a separate detailed
  `analysis/.../report.md` unless the user explicitly asks for an additional
  standalone report.

## Training Logging

- Formal model training must not depend on WandB or other remote experiment
  trackers for basic evidence. Use local CSV/JSON/Markdown artifacts by default.
- All formal training, evaluation, generation, and RL rollout jobs must write a
  durable local `.log` file under the run/output directory. Logs are the primary
  chronological record; CSV/JSON/SDF/Markdown artifacts supplement them but do
  not replace them.
- Use Python `logging` in core code instead of ad-hoc `print()`. Log lines should
  be readable and grep-friendly, with timestamp, level, rank/local-rank when
  distributed, module/component name, epoch or global step when available, and a
  concise message.
- Every formal training run must write scalar metrics to a local file under the
  run directory, such as `metrics/metrics.csv`, including train/val total loss
  and available component losses or accuracies.
- Terminal/stdout must show clear rank-0 progress during training: epoch,
  global step, elapsed time, total loss, and key component metrics. Shell launch
  scripts should redirect stdout/stderr to a durable `.log` file so the terminal
  progress is preserved.
- Terminal output should be human-readable during long runs: rank-0 only for
  progress summaries, no multi-rank progress-bar spam, no unstructured tensor
  dumps, and enough context to tell whether the run is loading data, sampling,
  scoring rewards, training PPO, validating, saving checkpoints, or shutting
  down.
- Each formal run log should include at least: config path and resolved config
  snapshot path, git commit/status summary, conda environment name, CUDA visible
  devices and GPU count, dependency/preflight results for nonstandard backends
  such as DiffGui, output directory, checkpoint save paths, validation summaries,
  warning/error tracebacks, and final completion/interruption status.
- Every formal training run must save a resumable checkpoint at the end of every
  completed epoch. Set checkpoint cadence to one epoch (for example,
  `checkpoint_every_epochs: 1`) and include model state, optimizer state, epoch,
  resolved config, and metrics. Formal launchers must preflight or reject configs
  that would leave any completed epoch without a checkpoint.
- WandB and other remote experiment trackers are disabled for formal training
  records unless the user explicitly requests a separate one-off exception; even
  then local records must remain complete.
- If a run predates this rule and lacks component metrics, state that limitation
  clearly before interpreting loss or model quality.

## Work Records

- For substantial coding, analysis, preprocessing, or experiment tasks, create a
  durable Markdown work record at the **absolute path**
  `/home/md666/data/disk6/DiffB110/work_records/<YYMM>/<YYMMDD>/`.
- Month-level grouping folder format: `<YYMM>/` (e.g. `26-6/` for June 2026).
  Daily folders go inside the month folder, e.g. `work_records/26-6/260609/`.
- Use one file per task, named `<YYMMDD>-<seq> <short Chinese title>.md`, for
  example `260601-14 为什么要先做数据构建.md`.
- The record should briefly cover motivation, relevant existing code/data,
  changes or commands, verification results, output files, and remaining risks.
- Daily/task reports must include detailed quantitative data when available:
  summary tables, key counts/rates/timings, and links to the exact generated
  CSV/JSON/SDF files. Figures and charts must be embedded directly in the
  report/record with Markdown image syntax such as
  `![caption](metric_by_method.png)`; do not provide only a file link or path.
- Images embedded from `work_records/` Markdown or archived HTML must be copied
  into the same directory as the document and referenced with a local filename
  such as `![caption](metric_by_method.png)` or `src="metric_by_method.png"`.
  Do not leave work-record image links pointing to `analysis/`, `OUT/`,
  `data/`, absolute paths, or parent directories; keep the work record
  self-contained for PDF export and archival movement. CSV/JSON/SDF evidence
  may still be linked to its source directory.

## Work Days (Daily Summaries & Reports)

- Daily individual `.md` summaries live under `/home/md666/data/disk6/DiffB110/work_days/`.
- Weekly merged PDF reports go to `work_days/每周/`.
- Monthly summary documents (when available) go to `work_days/每月/`.
- Conversion scripts (e.g. `md-pdf.py`) stay at `work_days/` root for reuse.
- **Naming format**: `谢东宸<YYMMDD>-<YYMMDD>[ 简短描述].ext`. No verbose labels
  such as "每日总结合并"; use "工作总结" or a brief distinguishing phrase only
  when multiple files share the same date range.

## Code Style

- Classes: `PascalCase`.
- Functions and variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Private helpers: `_leading_underscore`.
- Reward classes end with `Reward`.
- Fact/dataclass containers end with `Facts`.
- Imports are ordered as standard library, third-party, project-local, with blank
  lines between groups.
- Public APIs need parameter and return type annotations.
- Prefer `from __future__ import annotations` and PEP 604 unions.
- Use `TYPE_CHECKING` guards for type-only imports that could create runtime
  cycles.
- In dataclasses, fields without defaults must appear before fields with
  defaults.
- Every `.py` file, class, and public method should have a concise docstring.

## Tensor And Runtime Rules

- Do not hardcode `.cuda()` or `.cpu()` in core logic.
- Infer `device` from input tensors and pass `device=device` when creating new
  tensors.
- Temporary global or object state changes must be restored with `try/finally`.
- Use exception chaining with `raise ... from exc` when wrapping errors.
- Avoid `print()` in core logic; use `logging.getLogger(__name__)`. Rank-0
  progress output is acceptable.

## Reward System Rules

- Rewards implement `BaseReward.calculate_rewards(DataProto) -> DataProto`.
- Import `DataProto` and `TensorDict` from `ddpo.runtime.protocol`.
- Failed molecules must receive explicit finite penalties, not `NaN` or `Inf`.
- Advantage normalization uses global z-score via `RewardNormalizer`; do not
  reintroduce the deprecated `group_index` field.
- Do not create independent molecule-building paths. Reuse the existing
  molecule semantics and `rdkit_valid` fields.
- Reward naming must stay consistent through the full chain:
  YAML name -> `reward_prefix` -> batch fields -> dataclass fields -> CSV columns
  -> terminal display name.
- Each reward component must have distinct `BatchStats`, `BatchFacts`,
  `EpochFacts`, and CSV columns. Do not merge different reward values into the
  same field.

When changing reward names or fields, check at least:

- `ddpo/rewards/factory.py`
- the reward implementation file
- `ddpo/runtime/train_helpers.py`
- `ddpo/runtime/recording/schemas.py`
- `ddpo/runtime/recording/writers/molecule_writer.py`
- `ddpo/runtime/recording/writers/metrics_writer.py`
- `ddpo/runtime/recording/run_recorder.py`

## Molecular Descriptor Monitoring

- Treat molecular descriptor drift as an early warning signal, often earlier
  than total reward drift.
- For generated molecules, monitor at least carbon fraction, hetero atom count,
  element diversity, logP, TPSA, ring count, aromatic rings, three-member rings,
  fused/complex rings, and validity.
- Descriptor metrics should be computed only for `rdkit_valid=True` molecules
  unless the task is explicitly about invalid molecule handling.
- Prefer pure helper functions for descriptor extraction and pass metrics through
  existing recording/extra-metrics mechanisms instead of adding invasive state.
- Compare generated distributions against ChEMBL baselines and top-reward
  deciles, not only global epoch averages.

## Parameter Change Checklist

When changing any function or method parameter, trace the whole data flow.

- Search all call sites and verify explicit arguments and `**kwargs` pass-through.
- For abstract/interface methods, search every implementation of the method.
- For wrappers/decorators, ensure new parameters are forwarded to the wrapped
  object.
- If the parameter comes from YAML, CLI, or environment variables, check config
  read, config pass-through, and final use.
- New optional parameters need defaults that preserve existing configs.

Typical high-risk methods include `BaseReward.calculate_rewards` and any
factory-built reward, rollout, or recording interface.

## Configuration

- Build components from YAML through the project factory/build helpers.
- Put defaults in Python code and expose behavior through config where useful.
- Do not hardcode fingerprint type, radius, bit count, batch field names, GPU
  count, or training paths.
- Fingerprint-related code should use `use_features` or explicit config options,
  with module-level defaults such as `DEFAULT_FINGERPRINT_RADIUS` and
  `DEFAULT_FINGERPRINT_BITS`.

## Validation After Code Changes

Run the narrowest meaningful checks first, then broaden as needed.

1. Syntax-check every modified `.py` file with `py_compile`.
2. Run relevant tests under `test/`.
3. For ddpo pipeline changes, run `bash launch/smoke.sh` unless no GPU is
   available or the change is purely data/documentation.
4. Check terminal output for tracebacks and field-name mismatches.
5. Check CSV/JSON/SDF outputs when writers or schemas changed.

`launch/smoke.sh` is the standard 1 GPU, 1 epoch, batch 4 DDP smoke test. It has
cleanup logic for torchrun process trees, port 29500, and GPU memory reporting.

## Git Commit Norms

- Commit by logical task, not by file.
- Use commit messages like `YYMMDD-N 简单中文描述`.
- Before committing, run `git status --short` and add all intended files.
- Do not add ignored/generated large outputs such as `OUT/`, `checkpoints/`,
  `bishe/`, `QADD-main/`, `DiffMeta-main/`, or `RL-Diffusion-main/`.

## Completion Summary

For substantial project tasks, summarize in Chinese with:

- motivation/background
- relevant existing code/data
- what changed
- commands run and results
- output files or key numbers
- next steps only when useful

- Start each substantial summary from the original reason/problem being solved.
- Keep summaries concise, data-first, and table-friendly: use numbers where
  possible, tables where they clarify comparison, and avoid vague praise or
  unsupported qualitative claims.

## Random Seed Norm

All molecule generation, benchmark cross-model evaluation, and sharded sampling
use `seed = gpu_id` (1-based).

| Scenario | Rule | Example |
|---|---|---|
| Multi-GPU sharded generation | per-shard `seed = gpu_id` (1-based) | GPU 1 -> seed 1, GPU 7 -> seed 7 |
| Single-GPU generation | `seed = 1` | `--seed 1` |
| Benchmark cross-model eval | all models use `seed = gpu_id` | aligns DiffGui/DiffSBDD/PMDM/GCDM-SBDD/GraphBP/Pocket2Mol |
| DiffGui PPO/DDPO training | set `diffgui_docking_batch_sweep.seed_mode: gpu_id`; each local rank uses 1-based GPU/shard id | rank 0 -> seed 1, rank 7 -> seed 8 |

Forbidden:
- Date-based seeds such as `20260625 + gpu_id * 101`
- `seed = 0`
- Different seed formulas per model that make cross-model comparisons invalid
- New DiffGui training configs that rely on date-based `seed_base` progression
  instead of `seed_mode: gpu_id`

Rationale: the formal benchmark `launch_benchmark.sh` already uses
`seed = gpu_id` for the four main models (DiffGui/DiffSBDD/PMDM/GCDM-SBDD);
GraphBP and Pocket2Mol were aligned to this convention on 2026-06-25.

## DiffGui PPO Data-Reuse Norm

For formal DiffGui PPO/DDPO training where each newly generated molecule should
contribute to only one policy update, set `train.ppo_epochs: 1`. This controls
how many PPO optimization passes reuse the current epoch's rollout batch; it is
not checkpoint or optimizer-state reuse. Checkpoint reuse is controlled by
`diffgui_model.checkpoint`.
