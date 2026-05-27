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

## 2. Sync launch scripts with YAML configs

Files:
- ~~`launch/run_6gpu.sh` references non-existent `6gpu_qed_sa.yaml`~~ → DONE: 改为 `6gpu_dock.yaml`
- `launch/7gpu_classifier_only.yaml` has no dedicated `.sh` (use `run.sh` with `CONFIG_PATH`)
- `launch/molskill_test.yaml` has no dedicated `.sh`

## 3. Clean up stale .cursor rules → DONE

`.cursor/rules/` and `.codebuddy/rules/` 已同步为 7 个文件，新增规则 §十六 保护。
两个目录均不删除，保持内容一致。
