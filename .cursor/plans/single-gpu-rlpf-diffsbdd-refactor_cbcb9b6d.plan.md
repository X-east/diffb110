---
name: single-gpu-rlpf-diffsbdd-refactor
overview: Remove the current Lightning/DDP training path as the active workflow and establish a clean single-GPU RL pipeline that integrates DiffSBDD sampling with the existing RLPF-verl_diffusion EDMActor update loop, while preserving reward, mask, diffusion, and recording semantics.
todos:
  - id: inspect-boundaries
    content: Inspect current Lightning and actor update ownership boundaries
    status: completed
  - id: design-single-entry
    content: Design clean single-GPU training entrypoint and loop
    status: completed
    dependencies:
      - inspect-boundaries
  - id: implement-single-pipeline
    content: Implement single-GPU RL pipeline using DiffSBDD + EDMActor
    status: in_progress
    dependencies:
      - design-single-entry
  - id: wire-recording-checkpoints
    content: Wire recording and checkpointing without Lightning trainer
    status: pending
    dependencies:
      - implement-single-pipeline
  - id: add-single-config-launcher
    content: Add single-GPU config and launcher
    status: pending
    dependencies:
      - implement-single-pipeline
  - id: validate-single-gpu
    content: Run single-GPU smoke validation and summarize risks
    status: pending
    dependencies:
      - wire-recording-checkpoints
      - add-single-config-launcher
isProject: false
---

# Goal

建立一个**单卡可稳定运行**的 DiffSBDD + RLPF-verl_diffusion 强化学习主线，暂时停止把 Lightning/DDP 作为主训练入口。

# Why this refactor

当前训练链路同时存在两套优化/调度语义：

1. `EDMActor` 自带一套 PPO/优化器更新流程
2. Lightning `Trainer` + DDP 又提供一套训练/进程/保存框架

这会导致：

- 优化器所有权不清晰
- backward / step 边界容易混乱
- 单卡与多卡语义不一致
- checkpoint / log / epoch 记录路径被框架耦合
- 调试复杂度远高于科学问题本身

# Refactor principles

- 先做**单卡主线**，确保采样→奖励→PPO 更新→记录→checkpoint 全部打通
- 不改 reward 语义
- 不改 diffusion timestep 语义
- 不改 mask / batch 语义
- 不引入 Ray
- 暂时不追求 DDP，只保留未来可迁移接口

# Target architecture

## Active path

新增或重建一个单卡入口，例如：

- `ddpo/main_diffsbdd_single_gpu.py`

其职责：

1. 读取 config
2. 设置 seed
3. 初始化 DiffSBDDModel
4. 初始化单卡 DataLoader / Rollout
5. 初始化 Rewarder / Filter
6. 初始化 `EDMActor`
7. 手写 epoch loop：
  - generate batch
  - rollout.generate_minibatch
  - annotate molecule semantics
  - filter
  - reward
  - actor.update_policy
  - record batch / epoch
  - save checkpoint

## Legacy path handling

- `ddpo/main_diffsbdd_lightning.py` 保留但标记为 legacy / deprecated
- 新脚本与新配置不再默认调用 Lightning
- 不立即删除 Lightning 文件，先停止其作为主路径

# Files likely involved

- `ddpo/main_diffsbdd_lightning.py`  （降级为 legacy）
- `ddpo/runtime/data_loader.py`
- `ddpo/runtime/rollout.py`
- `ddpo/rewards/factory.py`
- `ddpo/trainer/lightning_ddpo_module.py` （保留，仅 legacy）
- `RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py`
- `ddpo/runtime/recording/*`
- 单卡启动脚本 / 单卡 config

# Implementation steps

1. 审查当前单卡可复用组件
2. 设计新的单卡主入口
3. 将当前 forward 内的 rollout/filter/reward/update 顺序迁移到手写 loop
4. 脱离 Lightning trainer/save/log 接口，改为直接调用 recorder
5. 为单卡入口提供独立 config 和 run script
6. 跑单卡 smoke test
7. 再考虑未来 DDP 迁移接口

# Risks

- actor.update_policy 当前默认假设某些 batch 结构来自 Lightning 路径
- checkpoint 保存逻辑目前与 Lightning callback 有耦合
- epoch / batch 记录逻辑可能默认 global rank 语义
- reward/failure logging 可能依赖现有 samples.batch 字段

# What to test

1. 单卡 1 epoch 采样 smoke test
2. 单卡 1 step reward smoke test
3. 单卡 PPO update smoke test
4. 单卡 molecules.csv/metrics.json/checkpoint 是否生成
5. 固定 seed 下可重复性
6. log-prob / loss finite 性检查

