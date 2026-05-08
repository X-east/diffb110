---
inclusion: always
---

# 项目上下文与规则

## 当前项目状态

**已验证的训练管线**：
- 8 卡 DDP：`torchrun` + `ddpo/main_diffsbdd_ddp.py` + `configs/qed_sa_docking_8gjl_8gpu.yaml`
- 单卡调试：`ddpo/main_diffsbdd_single_gpu.py` + `configs/qed_sa_1gpu.yaml`
- Lightning 已弃用

**对接后端**：
- 当前主后端：**UniDock**（8GJL 8卡配置）
- 备选：AutoDock-GPU, Smina
- Ray 已完全弃用

**奖励系统**：
- UniDock 对接 + QED + SA，组合加权
- z-score 归一化（RewardNormalizer）
- KL 散度约束已接入 Actor（kl_coef 可配，当前默认 0.0）

**代码架构**：
- Docking 模块 4 层解耦（docking → preparation → config → backends → registry）
- molecule_semantics.py 统一分子管线
- prepare_mol_for_docking() 加氢 + Gasteiger 电荷
- runtime/build.py 配置驱动工厂

## 修改代码前必读

- `docs/environment_info.md` — 环境
- `docs/project_paths.md` — 路径
- `docs/workflow_commands.md` — 工作流
- `docs/project_context.md` — 分子语义与奖励语义

## 规则

- 不假设路径，使用 docs 中的规范路径
- 不引入 Ray
- 使用 GPU 直连 docking/scoring
- 不静默改变奖励语义
- 不静默改变扩散 timestep 语义、mask 语义或 batch 语义
- 文档与代码冲突时，显式报告冲突

## Sudo 与权限规则

- 命令因权限失败时立即停止，提供精确 `sudo` 命令
- 不尝试绕过权限问题
- 执行 sudo 后验证结果

## 版本标签

- `[2026-03-14 23:00] autodockgpu-rl-test-pass` — AutoDock-GPU + Lightning DDP RL 端到端测试通过
- `[2026-03-15 20:00] tmux-training-workflow` — 确立 tmux 远程训练标准
- `[2026-05-08 00:00] codebase-restructure` — 大规模解耦重构：docking 4层分离、分子语义统一、KL 接入、加氢+Gasteiger 管线、配置驱动工厂

## 训练工作流 — tmux 标准

**远程训练默认使用 tmux**，防止 SSH 断线导致训练中断。

```bash
# 1. 新建 tmux 会话
tmux new -s train

# 2. 在 tmux 里启动训练
conda activate dif
bash scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh

# 3. 脱离会话（训练继续）
# 按 Ctrl+B，然后按 D
```

监控：`tmux attach -t train` / `tail -f OUT/<run_dir>/logs/training_*.log` / `watch -n 2 nvidia-smi`

## 对接后端状态

| 后端 | 状态 | 说明 |
|------|------|------|
| UniDock | **当前主后端** | 8GJL 8卡配置使用，conda 安装 |
| AutoDock-GPU | 备选 | 需要 binary + AutoGrid map |
| Smina | CPU 备用 | 无 GPU 依赖 |

- 所有 YAML 配置中 `reward.docking.backend` 指定后端
- `qed_sa_docking_8gjl_8gpu.yaml` 使用 `unidock`
- AutoDock-GPU 需要 GPU 访问，IDE shell 无 `/dev/nvidia*` 权限

## 输出目录结构

```
OUT/<run_name>_<日期>_<序号>/
├── config.yaml                  # 训练配置快照
├── training_summary.csv         # 每 epoch 汇总
├── epochs/                      # 每 epoch 详细记录
│   └── epoch_NNNN/
│       ├── metrics.json
│       └── molecules.csv
├── logs/                        # 训练日志
└── checkpoints/                 # 模型权重
```

## 分析文档规则

所有 AI 生成的分析/总结 markdown 文件统一存放到 `something/` 目录，命名格式：`YYYY-MM-DD-中文题目.md`
