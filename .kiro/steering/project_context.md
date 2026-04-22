---
inclusion: always
---

Before modifying code related to environment setup, file paths, training, sampling, docking, scoring, or rewards, first read:
- `docs/environment_info.md`
- `docs/project_paths.md`
- `docs/workflow_commands.md`
- `docs/project_context.md`

Rules:
- Do not assume paths.
- Use canonical paths from docs.
- Do not introduce Ray.
- Assume GPU-based docking/scoring.
- Do not silently change reward semantics.
- Do not silently change diffusion timestep semantics, mask semantics, or batch semantics.
- If docs and code conflict, report the conflict explicitly.

## Permission and Sudo Rules

- Do NOT silently skip steps that fail due to permission errors.
- When a command fails with "Permission denied", immediately stop and ask the user to run the specific command with `sudo`.
- Provide the exact `sudo` command for the user to copy-paste, for example:
  ```bash
  sudo cp /source/file /dest/file
  sudo ln -sf /source /dest
  sudo chmod +x /path/to/script
  ```
- Do not attempt workarounds (e.g., writing to ~/bin) without first asking the user if sudo is available.
- After asking the user to run a sudo command, poll every 5 seconds (using a shell check command) to verify the result before proceeding to the next step. Do not assume success — confirm it by checking the filesystem or running a verification command. Only continue once the check passes.

## Version Tags

Each tag records a milestone with timestamp and Chinese summary.
Format: `[YYYY-MM-DD HH:MM] tag-name — 中文简介`

- `[2026-03-14 23:00] autodockgpu-rl-test-pass` — AutoDock-GPU 与 Lightning DDP 强化学习（DDPO）端到端测试通过，4 GPU 对接奖励正常计算，得分 mean=-4.005 kcal/mol，valid=10/16。
- `[2026-03-15 20:00] tmux-training-workflow` — 确立 tmux 为远程训练标准工作流，修复 NCCL ALLGATHER 超时崩溃（all_gather 从 forward 移至 training_step），修复奖励汇总重复打印，统一 batch 语义。

## Training Workflow — tmux Standard

**Ubuntu 24 远程训练默认使用 tmux。** 这是防止 SSH 断线/IDE 重载导致训练中断的唯一可靠方式。

### 启动训练（标准步骤）
```bash
# 1. 新建 tmux 会话
tmux new -s train

# 2. 在 tmux 里启动训练
bash run_4gpu_test.sh

# 3. 脱离会话（训练继续在后台跑）
# 按 Ctrl+B，然后按 D
```

### 监控训练
```bash
# 重连 tmux 会话
tmux attach -t train

# 查看所有会话
tmux ls

# 实时查看日志（在另一个终端）
tail -f ./OUT/<run_dir>/logs/training_*.log

# 监控 GPU
watch -n 2 nvidia-smi
```

### 规则
- 不要在裸 SSH 终端或 IDE 内置终端里直接 `bash run_xxx.sh` 运行长时间训练
- 所有训练脚本都应支持在 tmux 会话里运行
- 不要依赖 IDE 终端窗口保持连接
- 训练日志必须写入文件（`logs/training_*.log`），不能只靠终端输出

## Output Directory Structure

每次训练的输出根目录为 `OUT/<run_name>_<日期>_<序号>/`，内部结构如下：

```
OUT/<run_name>_<日期>_<序号>/
├── config.yaml                  # 本次训练完整配置快照
├── training_summary.csv         # 每 epoch 一行的汇总文档（奖励/loss/有效率/时间）
├── checkpoints/                 # Lightning ModelCheckpoint 保存的模型权重
│   ├── diffsbdd-ddpo-epoch=04-train_loss=0.0007.ckpt
│   ├── diffsbdd-ddpo-epoch=09-train_loss=0.0006.ckpt
│   └── last.ckpt                # 最后一个 epoch 的权重
├── epochs/                      # 每个 epoch 的详细记录
│   ├── epoch_0001/
│   │   ├── metrics.json         # 该 epoch 的奖励/loss/有效率等指标
│   │   └── molecules.csv        # 该 epoch rank-0 本地分子的 reward/qed/sa
│   ├── epoch_0002/
│   │   └── ...
│   └── epoch_NNNN/
├── logs/
│   ├── training_<timestamp>.log # 完整训练日志（含所有 epoch 的汇总）
│   ├── molecules_<timestamp>.csv  # BatchRecorder 分子记录
│   └── batch_stats_<timestamp>.csv # BatchRecorder batch 统计
└── lightning_logs/
    └── version_0/               # TensorBoard 事件文件
        └── events.out.tfevents.*
```

### training_summary.csv 字段
| 字段 | 说明 |
|---|---|
| epoch | epoch 编号（1-indexed）|
| timestamp | 该 epoch 完成时间 |
| avg_loss | epoch 平均 loss |
| ppo_loss | PPO loss |
| clip_frac | PPO clip fraction |
| valid_g | 全局有效分子数 |
| total_g | 全局总分子数 |
| val_pct | 有效率 |
| rw_mean/min/max | 全局奖励均值/最小/最大 |
| qed_mean | QED 均值（含无效=0）|
| sa_mean | SA 均值（含无效=0）|
| dock_mean | Docking 奖励均值（如启用）|
| drug_mean | Druglikeness 奖励均值 |

### 规则
- 每次训练生成独立目录，不覆盖历史
- `epochs/` 子目录按 epoch 编号（4位补零）命名
- `training_summary.csv` 追加写入，每 epoch 一行，训练中断后重启可断点续写
- checkpoint 保存间隔由 `train.checkpoint_interval`（默认5）控制

## Current Codebase Status

The repository still contains multiple Ray-dependent scripts, launchers, and dependency checks.
However, the intended current project direction is GPU-based docking/scoring without introducing new Ray-based orchestration.

When modifying training or docking code:
- do not introduce new Ray dependencies
- treat existing Ray-based scripts as legacy unless explicitly required
- prefer the Lightning-based multi-GPU training path centered on `ddpo/main_diffsbdd_lightning.py`
- prefer the user-maintained bash launch scripts as the day-to-day runtime interface
- keep command-line arguments consistent with the verified `--help` output

## Docking Backend Status

- Primary docking backend: **AutoDock-GPU** (`autodock_gpu`)
- Binary compiled at: `/home/md666/data/disk6/DiffB110/AutoDock-GPU/AutoDock-GPU-develop/bin/autodock_gpu_64wi`
- Binary symlinked to: `/usr/local/bin/autodock_gpu_64wi` (on PATH)
- AutoGrid4 maps generated and stored at: `data/8GJJ-C-Cip_charged.maps.fld`
- Charged receptor PDBQT: `data/8GJJ-C-Cip_charged.pdbqt`
- MGLTools installed at: `/opt/mgltools_x86_64Linux2_1.5.7`
- AutoGrid4 binary: `/opt/x86_64Linux2/autogrid4` (symlinked to `/usr/local/bin/autogrid4`)
- Smoke test result: reference ligand `cip.pdbqt` docked at **-4.91 kcal/mol** (Estimated Free Energy of Binding), best inter+intra **-9.11 kcal/mol**
- UniDock (`unidock`) is legacy / broken — do not revert to it
- All YAML configs use `backend: "autodock_gpu"` and `fld_file: "./data/8GJJ-C-Cip_charged.maps.fld"`
- NOTE: AutoDock-GPU requires GPU access. The Cursor IDE shell runs as root without `/dev/nvidia*` access — always run training and docking from a normal user terminal.

### AutoDock-GPU Device Numbering

- AutoDock-GPU `--devnum` is **1-indexed**: `--devnum 1` = first GPU (GPU 0 in nvidia-smi)
- `nvidia-smi` and CUDA (`CUDA_VISIBLE_DEVICES`) are **0-indexed**
- Conversion in code: `self._devnum = gpu_id + 1` where `gpu_id` is 0-indexed
- Example: to use nvidia-smi GPU 2, set `gpu_id=2` → `--devnum 3`
- Do NOT set `CUDA_VISIBLE_DEVICES` when using AutoDock-GPU, as it renumbers devices and breaks the `--devnum` mapping

## Analysis and Summary Documents Rule

所有由 AI 生成的项目分析文档、训练结果总结、数据分析报告，必须统一存放到项目根目录下的 `something/` 文件夹内。

### 命名规范
- 格式：`YYYY-MM-DD-中文题目.md`
- 示例：`2026-03-19-训练第一阶段druglikeness奖励分析.md`
- 日期使用当天实际日期（今天是 2026-03-19）
- 题目用中文，简明扼要描述文档内容

### 规则
- 所有分析/总结 markdown 文件只写入 `something/` 目录，不放在项目根目录或其他位置
- 如果 `something/` 目录不存在，在写入前先创建它
- 不得使用英文文件名（日期部分除外）
- 每份文档独立命名，不覆盖已有文档