# DiffB110 Codex Project Guide

## Project Scope

This repository is the DiffB110 project: DiffSBDD generation for the RE-CmeB
target, optimized with DDPO/PPO and composite molecular rewards. Treat
`ddpo/main_diffsbdd_ddp.py` as the current training entry point.

Before changing project code, read:

- `docs/项目情况快速总览.md`
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
- Reference generated plots and data files from the Markdown report with
  relative links to files in the same analysis directory, so rendered previews
  show figures inline and readers can open the exact CSV/JSON/SDF evidence.
- Plot filenames should be descriptive and stable, such as
  `global_distribution_comparison.png`, `run_drift_summary.png`, or
  `top_reward_failure_modes.png`.

## Work Records

- For substantial coding, analysis, preprocessing, or experiment tasks, create a
  durable Markdown work record under `work_records/<YYMMDD>/`.
- Use one file per task, named `<YYMMDD>-<seq> <short Chinese title>.md`, for
  example `260601-14 为什么要先做数据构建.md`.
- The record should briefly cover motivation, relevant existing code/data,
  changes or commands, verification results, output files, and remaining risks.
- Daily/task reports must include detailed quantitative data when available:
  summary tables, key counts/rates/timings, and links to the exact generated
  CSV/JSON/SDF files. When plots are useful, save them beside the report and
  embed them with Markdown relative links such as
  `![caption](metric_by_method.png)`.

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
