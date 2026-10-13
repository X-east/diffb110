---
inclusion: always
---

# 项目上下文与规则

## 当前项目状态

DiffB110 当前恢复为并行路线组织，不再确认某一条正式主线。并行方向包括 DiffSBDD 原始/轨迹分支、BC/AWR/IQL 离线实验、AliDiff 风格 Global DPO、DecompDPO 风格局部偏好优化、在线 DiffSBDD DDPO 和 DiffGui PPO/DDPO。各路线共享横评、固定尝试次数、分子重建、筛选和 MD 交接规范，但模型、损失、数据和验收结果分别记录。

## 修改代码前必读

- `docs/project_context.md` — 当前项目方向、分子语义、奖励语义和路线决策
- `docs/project_paths.md` — 路径和输出根目录
- `docs/workflow_commands.md` — 工作流、训练、生成、筛选和 MD 交接说明
- `AGENTS.md` — 仓库规则

## 当前主入口

- DiffSBDD 完整轨迹采集：`ddpo/main_diffsbdd_collect_ddp.py`
- 并行路线记录：`work_records/26-7/260731/`、`work_records/26-8/260822/`、`reference project/AliDiff-main/`、`reference project/DecompDPO/`
- 训练/生成入口：`ddpo/main_diffgui_docking_ddp.py`、`ddpo/main_diffsbdd_ddp.py`、`ddpo/main_diffsbdd_collect_ddp.py`
- Active launch configs：`launch/`
- Route outputs：`OUT/`
- Analysis evidence：`analysis/`
- Work records：`work_records/`

## 规则

- 不假设路径，使用 docs 中的规范路径。
- 不引入 Ray。
- DiffGui 模型加载、生成和 PPO/DDPO 使用 `diffgui` 环境。
- DiffSBDD、reward scoring、analysis 和多数测试使用 `dif` 环境。
- 不静默改变奖励语义。
- 不静默改变扩散 timestep 语义、mask 语义或 batch 语义。
- 横评和生成默认按固定尝试次数与固定 seed 规则比较；不要通过反复重采样补齐无效分子。
- 文档与代码冲突时，显式报告冲突。

## 并行路线共享工作流

```text
cross-model diffusion benchmark evidence
  -> route-specific data, trajectories or preference pairs
  -> parallel training and ablation
  -> fresh generation from each checkpoint
  -> RDKit / docking / LE / QED-SA / ChEMBL-like / structural pathology screening
  -> top candidate molecules
  -> molecular dynamics simulation
```

MD 模拟是筛选后的下游验证阶段。正式开始前必须记录候选分子来源、对接构象、蛋白/配体准备方式、力场、模拟时长、重复数和接受标准。

## 版本标签

- `[2026-03-14 23:00] autodockgpu-rl-test-pass` — AutoDock-GPU + Lightning DDP RL 端到端测试通过。
- `[2026-03-15 20:00] tmux-training-workflow` — 确立 tmux 远程训练标准。
- `[2026-06-24 10:00] diffgui-rl-mainline` — 历史路线：DiffGui RL 曾作为主线；保留其结果和运行记录。
- `[2026-08-17 10:00] diffsbdd-offline-iql-mainline` — 历史路线切换记录；当时曾将 DiffSBDD 离线 IQL 设为候选主线。
- `[2026-08-24 10:00] parallel-routes-restored` — 撤回单一主线确认，恢复 DiffSBDD、AliDiff/Global DPO、DecompDPO/Local DPO、IQL/BC/AWR 与在线 PPO/DDPO 的并行实验组织。
