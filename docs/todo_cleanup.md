# TODO Cleanup

> **最后更新**: 2026-05-27

## 1. Resolve PyTorch version compatibility

Current environment:
- `torch==2.0.1+cu118`

Observed warning:
- one imported component expects `PyTorch >= 2.2`

Need:
- identify which dependency emits the warning
- decide whether to keep current environment or upgrade

## 2. Resolve Uni-Dock local-source import conflict

Problem:
- UniDock is installed as a conda package, not a local repo copy
- Previous `Uni-Dock/` directory at project root has been removed

Status:
- Current reward pipeline uses conda-installed `unidock_tools` via Python import, works correctly
- No action needed unless local-source workflow is required

## 3. Fix launch script / YAML mismatches

Files:
- `launch/run_6gpu.sh` references `launch/6gpu_qed_sa.yaml` which does not exist
- `launch/7gpu_classifier_only.yaml` has no dedicated `.sh` (use `run.sh` with `CONFIG_PATH`)
- `launch/molskill_test.yaml` has no dedicated `.sh`

## 4. Clean up stale .cursor rules

Directory `.cursor/rules/` contains old rules predating the move to `.codebuddy/rules/`.
Consider consolidating into `.codebuddy/rules/` or deleting stale `.cursor/` directory.
