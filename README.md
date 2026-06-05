# DiffB110: DiffSBDD-DDPO 分子生成与优化

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **基于 DiffSBDD 的口袋特异性 3D 分子生成、模型内部改造与 DDPO 强化学习优化**

---

## 项目概述

本项目面向 RE-CmeB / CmeB 口袋，围绕 DiffSBDD 进行 3D 分子生成、模型内部质量改造和 DDPO/PPO 强化学习优化。核心目标是生成具有潜在结合能力、化学结构合理、可继续筛选和优化的候选分子。

当前项目分为三条并行路线：

1. **小改路线**：保留 DiffSBDD 生成主链，在模型内部加入 atom prior、x0 geometry/bond length loss、bond-aware auxiliary head 等轻量增强，暂不引入完整 edge diffusion。
2. **大改路线**：把 DiffSBDD 本体升级为可选 DiffSBDD-xhe 模式，在原有 `x/h` 扩散链中加入 bond type / halfedge 的 `e` 扩散；standalone edge diffusion 只作为隔离验证工具，不作为最终替代底座。
3. **强化学习路线**：继续维护 DDPO/PPO 主线，升级 reward function，引入 docking、classifier、QED/SA、化学几何护栏和分布约束；高分且结构合理的分子会进入候选池，用于 replay buffer、自训练或偏好微调。

### 已完成核心能力

- **DDP 多卡训练** — `torchrun` + `main_diffsbdd_ddp.py`，8卡生产环境已验证
- **UniDock/QED/SA 奖励** — 完整的 Docking + 类药性多目标奖励管线
- **高维 ChEMBL classifier reward** — 2220 维分子特征 MLP 奖励已接入
- **模型质量诊断与多模型横评** — 已对 DiffSBDD、DiffGui、DePPA、PMDM、DiffInt、DiffPharma 在 8GJL/C1101 上进行统一生成与 docking 对比
- **DiffSBDD-xhe 数据链基础** — 已建立 halfedge 数据、oracle reconstruction、edge semantic bundle、standalone edge denoiser/DDP/sampler 验证闭环，下一步迁移进 DiffSBDD 本体
- **KL 散度约束** — 已接入 Actor loss，`kl_coef` 可配置（当前默认 0.0，未实际测试）
- **代码大规模解耦重构** — Docking 模块 4 层分离、分子语义管线统一、加氢+Gasteiger 电荷
- **配置驱动** — YAML 配置 + 工厂模式构建全部组件

---

## 项目结构

```
DiffB110/
├── ddpo/                              # 核心集成代码（训练主线）
│   ├── main_diffsbdd_ddp.py           # 多卡 DDP 训练入口（生产用）
│   ├── main_diffsbdd_single_gpu.py    # 单卡调试入口
│   ├── main_diffsbdd.py               # 旧入口（含 Ray 逻辑，legacy）
│   ├── integration/                   # DiffSBDD 适配层
│   │   ├── diffsbdd_model.py          # DiffSBDDModel 包装器
│   │   └── mask_utils.py              # mask 格式转换
│   ├── kl/                            # KL 散度约束
│   │   └── gaussian_kl.py             # 高斯 KL 计算纯函数
│   ├── rewards/                       # 奖励系统（4层解耦）
│   │   ├── docking.py                 # 对接奖励语义层
│   │   ├── docking_preparation.py     # SDF 输入准备 + 加氢 + Gasteiger
│   │   ├── docking_config.py          # 配置常量
│   │   ├── docking_registry.py        # 后端注册与选择
│   │   ├── docking_factory.py         # 工厂函数
│   │   ├── docking_backends/          # 对接后端实现
│   │   │   ├── unidock.py             # UniDock（当前主后端）
│   │   │   ├── autodock_gpu.py        # AutoDock-GPU（备选）
│   │   │   └── smina.py               # Smina（CPU 备用）
│   │   ├── interfaces/                # 奖励抽象接口
│   │   │   ├── base_reward.py
│   │   │   ├── docking_backend.py
│   │   │   └── docking_result.py
│   │   ├── normalization.py           # 奖励 z-score 归一化
│   │   ├── composite.py               # 组合奖励
│   │   ├── qed_reward.py              # QED 奖励
│   │   ├── sa_reward.py               # SA 奖励
│   │   └── stability.py               # 稳定性奖励
│   ├── runtime/                       # 运行时基础设施
│   │   ├── build.py                   # 工厂：从 YAML 构建全部组件
│   │   ├── rollout.py                 # 轨迹采样
│   │   ├── recording/                 # 训练记录（run_recorder, schemas, writers）
│   │   ├── data_loader.py             # 数据加载
│   │   ├── distributed.py            # DDP 通信
│   │   ├── seed.py                    # 随机种子管理
│   │   └── train_helpers.py           # 训练辅助
│   ├── utils/                         # 工具模块
│   │   ├── molecule_semantics.py      # 统一分子构建+过滤+加氢+电荷
│   │   ├── diffsbdd_filter.py         # 有效性过滤
│   │   ├── molecule_recorder.py       # 分子级 CSV 记录
│   │   ├── batch_recorder.py          # batch 级统计记录
│   │   └── logger.py                  # 日志工具
│   └── patches/                       # 历史补丁
├── DiffSBDD/                          # 原始扩散模型工程（READ-ONLY）
├── RLPF-verl_diffusion/               # RL 框架参考代码
│   └── verl_diffusion/worker/actor/edm_actor.py  # EDMActor（PPO 策略更新）
├── configs/                           # YAML 训练配置
│   ├── qed_sa_docking_8gjl_8gpu.yaml  # 当前主生产线配置（8GJL + UniDock）
│   ├── qed_sa_1gpu.yaml              # 单卡调试配置
│   ├── qed_sa_2gpu.yaml              # 2 卡配置
│   └── qed_sa_8gpu_16mol_50ep.yaml   # 8 卡无对接配置
├── data/                              # 受体、配体、AutoGrid map
│   ├── 8gjl.pdbqt                    # 当前 8GJL 工作流受体
│   └── 8GJJ-C-Cip_charged.*          # 8GJJ 工作流受体 + map
├── scripts/                           # 启动脚本
│   └── run_rl_qed_sa_docking_8gjl_8gpu.sh  # 当前 8 卡生产启动脚本
├── test/                              # 测试
├── materials/                         # 设计笔记（流形奖励/判别器方案）
└── OUT/                               # 训练输出
```

---

## 技术路线

```
                    ┌──────────────────────────────────┐
                    │ DiffSBDD 快速生成主线             │
                    └──────────────────────────────────┘
                                  ↓
        ┌─────────────────────────┼─────────────────────────┐
        ↓                         ↓                         ↓
 小改路线                  大改路线                  强化学习路线
 atom prior               edge diffusion             DDPO/PPO
 geometry loss            x/h/e denoiser             docking/classifier
 bond-aware head          DiffSBDD-xhe mode           chemistry guards
        ↓                         ↓                         ↓
 短期改善质量              长期升级 DiffSBDD          高分好分子回流训练
```

三条路线保持解耦：

- 小改路线优先服务六个月内稳定产出。
- 大改路线以可选 `DiffSBDD-xhe` 模式推进，standalone edge diffusion 仅作为验证平台，不破坏当前可运行 RL 主线。
- 强化学习路线负责 reward、筛选、top-N/replay/self-training，但不盲目用高分分子直接重训。

### 训练入口

| 入口 | 用途 | 命令 |
|------|------|------|
| `scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh` | 8 卡生产 | `bash scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh` |
| `ddpo/main_diffsbdd_ddp.py` | 多卡 DDP | `torchrun --nproc_per_node=8 ddpo/main_diffsbdd_ddp.py` |
| `ddpo/main_diffsbdd_single_gpu.py` | 单卡调试 | `python ddpo/main_diffsbdd_single_gpu.py --config_path configs/qed_sa_1gpu.yaml` |

---

## 当前状态

### 已完成

- [x] DDP 8 卡训练管线（torchrun + main_diffsbdd_ddp.py）
- [x] UniDock 对接后端 + AutoDock-GPU/Smina 备选
- [x] QED + SA 类药性奖励
- [x] 奖励 z-score 归一化（RewardNormalizer）
- [x] KL 散度约束接入 edm_actor.py（kl_coef 可配置）
- [x] Docking 模块 4 层解耦（docking → preparation → config → backends → registry）
- [x] 分子语义统一（molecule_semantics.py + prepare_mol_for_docking）
- [x] 加氢 + Gasteiger 电荷管线
- [x] 配置驱动工厂模式（runtime/build.py）
- [x] 训练记录系统（recording/）
- [x] 参考模型深拷贝（ref_model for KL）

### 未完成验证项

- [ ] KL 散度约束实际测试（当前 kl_coef=0.0）
- [ ] 加氢 + Gasteiger 电荷对 docking 效果的真实训练验证
- [ ] `compute_gaussian_kl()` NaN/Inf 防护
- [ ] KL 分离日志（当前只有混合 loss，无法单独监控 KL）
- [ ] 奖励漂移（reward drift）治理

### 下一阶段研究目标

1. **小改路线** — atom composition prior、x0 geometry/bond length loss、bond-aware auxiliary head。
2. **大改路线** — 将 halfedge 数据链、edge transition、edge head、sampler、predicted-edge decoder 迁移进 DiffSBDD 本体，形成可选 DiffSBDD-xhe 模式，并在 supervised generation 稳定后再考虑 RL logprob 接入。
3. **强化学习路线** — docking/classifier + 化学几何护栏，保存高分合法分子用于 replay buffer、自训练或偏好微调。
4. **评价体系** — 对生成分子统一计算 validity、ring、atom composition、bad bond length、docking、classifier、top-N pose 检查。

---

## 成功指标

| 指标 | 目标 |
|------|------|
| Docking Score | < -8 kcal/mol |
| 分子有效性 | > 80% |
| QED | > 0.5 |
| SA Score | < 4 |

---

**Last Updated**: 2026-06-04
