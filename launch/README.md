# launch 目录说明

> 最后更新：2026-06-30

`launch/` 是训练、smoke 和实验矩阵的入口目录。当前项目第一部分是“扩散模型横评选择 DiffGui -> DiffGui 强化学习 -> RL 后模型生成与筛选 -> 最优候选进入分子动力学模拟”。其中本目录主要负责 DiffGui 强化学习训练、smoke、续训和矩阵实验入口；RL 后生成、筛选和 MD 交接证据应记录到 `OUT/`、`analysis/` 和 `work_records/`。

DiffSBDD 训练入口保留为对照和备用路线。根目录只保留当前活跃入口；历史脚本、研究线配置、自动生成配置和运行日志统一放在 `launch/archive/`。

## 推荐入口

| 类型 | 入口 | 说明 |
|---|---|---|
| DiffGui DDP smoke | `bash launch/run_diffgui_docking_train_ddp_smoke.sh` | 当前主线 smoke，需要 `diffgui` 环境 |
| DiffGui DePPA-style 正式训练 | `bash launch/run_diffgui_deppa_last3_lr1e5_b18x8_t200_100ep_20260623.sh` | 当前 DiffGui RL 正式训练入口 |
| DiffGui next6 矩阵 | `bash launch/run_diffgui_next6_matrix_20260618.sh` | 串行矩阵复现和后续对照入口 |
| DiffGui 矩阵 | `bash launch/run_diffgui_ring_control_matrix_20260615.sh` | 当前 ring/control reward 矩阵复现入口 |
| DiffSBDD smoke | `bash launch/smoke.sh` | 保留线标准 1 GPU smoke |
| DiffSBDD 保留训练 | `CONFIG_PATH=launch/<config>.yaml bash launch/run.sh` | DiffSBDD 对照/备用 DDP 入口 |
| DiffSBDD 单奖励矩阵 | `bash launch/run_diffsbdd_single_reward_matrix_20260612.sh` | 保留线单 reward ablation 矩阵复现入口 |

## 核心配置

| 配置 | 用途 |
|---|---|
| `diffgui_deppa_last3_lr1e5_b18x8_t200_100ep_20260623.yaml` | 当前 DiffGui RL DePPA-style 正式配置 |
| `diffgui_next4_continue_to50_20260621.yaml` | DiffGui next4 续训到 50 epoch 配置 |
| `smoke.yaml` | DiffSBDD 标准 smoke |
| `7gpu_dock_classifier_v0_1.yaml` | docking + ChEMBL classifier 正式配置 |
| `8gpu_dock.yaml` | docking + ChEMBL prior / descriptor 配置 |
| `8gpu_chembl_only.yaml` | ChEMBL/prior 类配置 |
| `diffgui_docking_train_ddp_smoke.yaml` | DiffGui DDP smoke |

## Archive 分层

这些文件有复现实验价值，但不作为新任务默认入口：

| 路径 | 内容 |
|---|---|
| `archive/legacy_diffsbdd_wrappers/` | 旧 6/8GPU wrapper 和被通用 `run.sh` 取代的配置 |
| `archive/diffgui_historical/` | DiffGui 单次历史实验脚本、旧矩阵和模板配置 |
| `archive/research_lines/` | anti-carbon guard、edge diffusion、x/h/e、MolSkill 等研究线配置 |
| `archive/generated/` | 自动生成矩阵 YAML、matrix CSV/JSON |
| `archive/logs/` | 矩阵 launcher 日志 |
| `archive/runtime_logs/` | 原先散落在 `launch/` 根目录的 nohup 日志 |
| `archive/cache/` | Python bytecode cache |

## 不建议继续作为新入口的脚本

以下 6/8GPU 专用 wrapper 已移入 `archive/legacy_diffsbdd_wrappers/`。它们与 `CONFIG_PATH=... bash launch/run.sh` 功能重叠，并且多处仍引用不存在的 `launch/yaml_launch_env.py`。除非先修复，否则不要用于新训练：

- `archive/legacy_diffsbdd_wrappers/run_6gpu.sh`
- `archive/legacy_diffsbdd_wrappers/run_6gpu_dock.sh`
- `archive/legacy_diffsbdd_wrappers/run_8gpu_dock.sh`
- `archive/legacy_diffsbdd_wrappers/run_6gpu_chembl_only.sh`
- `archive/legacy_diffsbdd_wrappers/run_8gpu_chembl_only.sh`
- `archive/legacy_diffsbdd_wrappers/run_6gpu_dock_kl_sweep.sh`

推荐替代：

```bash
CONFIG_PATH=launch/8gpu_dock.yaml bash launch/run.sh
CONFIG_PATH=launch/8gpu_chembl_only.yaml bash launch/run.sh
```

## 不属于入口层的内容

| 路径 | 处理原则 |
|---|---|
| `launch/archive/generated/` | 自动生成配置；应由 generator 复现，不作为手写入口维护 |
| `launch/archive/logs/` | 矩阵 launcher 日志；属于运行证据 |
| `launch/archive/runtime_logs/` | 原顶层 nohup 日志 |
| `launch/archive/cache/` | Python bytecode cache，可清理 |

完整审计见：

- `analysis/launch_cleanup_20260618/report.md`
- `analysis/launch_cleanup_20260618/launch_inventory.csv`
