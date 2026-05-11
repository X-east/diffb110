# TODO Cleanup

## 1. Fix legacy CLI argument mismatch
- File: `scripts/train_unidock_8gpu.sh`
- Problem: uses `--config`
- Verified CLI of `ddpo/main_diffsbdd.py` expects `--config_path`

## 2. Mark Ray-based scripts as legacy
Files:
- `scripts/train_8gpu.sh`
- `scripts/train_unidock_8gpu.sh`
- `ddpo/configs/run_training.sh`
- `ddpo/configs/run_training_ddp.sh`

Reason:
- current intended direction is GPU-based docking/scoring without introducing new Ray orchestration

## 3. Resolve Uni-Dock local-source import conflict
Problem:
- local repository source conflicts with installed `unidock_tools` package in environment

Need:
- define a stable local-source execution method
- likely use `PYTHONPATH` or module-based invocation from local source tree

## 4. Review PyTorch version compatibility
Current environment:
- `torch==2.0.1+cu118`

Observed warning:
- one imported component expects `PyTorch >= 2.2`
Need:
- identify which dependency emits the warning
- decide whether to keep current environment or upgrade
