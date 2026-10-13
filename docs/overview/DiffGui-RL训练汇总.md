# DiffGui 强化学习历次主要训练汇总

纳入主要训练 20 个，其中当前核心是 6 个 `diffgui-next6` 20 epoch 组合矩阵；另保留 8 个单奖励归因 run、2 个旧组合 run、4 个早期正式/短训背景 run。

## 当前采用：DiffGui DePPA full **完整 100 轮**

> 最后更新：2026-09-29。以下为**一次完整的 100 轮训练**（`diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep`），
> 不再区分中间阶段；**采用 ep79（对接最优）**。

| checkpoint | n | 对接均值 | **含三元环%** | 螺环% | 桥环% | 环数 | 芳香环 | 可旋转键 | 重原子数 | R1–R4 命中 | R1–R5 命中 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ep1（RL 起点） | 57 | −9.78 | 12.3% | 12.3% | 14.0% | 3.77 | 1.72 | 0.51 | 18.19 | 68.4% | 71.9% |
| **ep79（现采用）** | 58 | **−11.28** | **37.9%** | **39.7%** | 24.1% | 5.10 | 1.19 | 0.57 | 21.14 | 65.5% | 69.0% |
| ep89（论文表 2.6 原用） | 61 | −10.63 | 44.3% | 27.9% | 19.7% | 4.26 | 1.18 | 0.95 | 18.97 | 62.3% | 70.5% |
| ep100 | 60 | −10.37 | 33.3% | 23.3% | 15.0% | 4.18 | 0.88 | 0.65 | 18.3 | 56.7% | — |

> **口径**：`structural_violation_count` = **R1–R4**（不含组成护栏 R5）；`violation_count` = **R1–R5**。
> 表中为**命中率**（至少违反一条）；ChEMBL 参照写的是**全清 80.94%**，命名不同，**不可直接横比**。

**采用 ep79 的理由**：对接为全程最优（−11.28，比 ep89 的 −10.63 好 0.65），且三元环（37.9% vs 44.3%）、
R1–R4 命中（65.5% vs 62.3%）均优于 ep89；代价是螺环率更高（39.7% vs 27.9%）、分子更大（HA 21.14 vs 18.97）。
ep100 结构最健康但对接最弱（−10.37），不采用。

**⚠️ 注意（与论文表述冲突）**：采用 ep79 后，“重原子数基本不变（18.2→19.0）”**不再成立**
（ep79 为 18.19→21.14，环数 3.77→5.10）——模型确实把分子推向更大更稠密的环系，论文该句已同步修改。

参照：ChEMBL 三元环 4.5%、螺环 2.6%、桥环 2.7%；`DiffGui ep1` 见上表首行。
即 RL 训练全程都在用“提高三元环/螺/桥环”换取对接改善——即本项目的 reward hacking 主线证据。

**分项命中率（供 §2.3.5）**：

| | R1 大环 | R2 环系 | R3 四元环 | R4a | **R4b** | **R4c** | R5 碳比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ep1 | 38.6% | 54.4% | 5.3% | 5.3% | 8.8% | 8.8% | 33.3% |
| **ep79** | 53.4% | 56.9% | 12.1% | 5.2% | **37.9%** | **37.9%** | 41.4% |
| ep89 | 47.5% | 47.5% | 1.6% | 1.6% | 29.5% | 27.9% | 29.5% |

## 目录
1. [早期正式大训练 / 背景短训](#早期正式大训练--背景短训)
2. [Reward 归因实验 / 单奖励矩阵](#reward-归因实验--单奖励矩阵)
3. [当前核心组合矩阵](#当前核心组合矩阵)
   - [旧组合奖励 / 环控制矩阵](#旧组合奖励--环控制矩阵)
   - [核心 next6 / 环病理组合矩阵](#核心-next6--环病理组合矩阵) 
4. [当前核心组合矩阵 / 旧组合奖励 / 环控制矩阵](#当前核心组合矩阵--旧组合奖励--环控制矩阵)

## Reward 组件含义速览

| reward 组件 | 内容 | 主要作用 |
| --- | --- | --- |
| aromatic_scaffold | 奖励单苯环；惩罚无苯环、过多苯环/杂环、融合环、三元环和卤素过多。 | 保留简单芳香骨架，同时压异常环形态。 |
| ring_pathology | 从基础分扣除环病理：环过多、大环、大环并六元环、三元环过多或 fused/spiro 三元环。 | 专门压 DiffGui 里出现的结构病理环。 |
| chemistry_guard | 检查碳比例、杂原子比例、三元环、碎片数、异常键长和异常卤素位置。 | 基础化学护栏，防止明显坏结构被继续放大。 |
| chembl_classifier | Morgan count + RDKit 2D 特征输入 MLP，输出 ChEMBL-like 概率并映射为奖励/惩罚。 | 把分子拉回更像 ChEMBL 的 drug-like 化学空间。 |
| ring_complexity | 软惩罚超过目标的总环数、芳香环数、融合环、三元环、螺原子和桥头原子。 | 旧版环复杂度控制；比 ring_pathology 更宽泛。 |

## 分组分析

### 1. 早期正式大训练 / 背景短训

![早期正式大训练 / 背景短训 训练曲线](../../work_records/26-6/260618/group_early_training_curves.png)

![早期正式大训练 / 背景短训 生成分子性质](../../work_records/26-6/260618/group_early_molecule_properties.png)

![早期正式大训练 / 背景短训 最终指标热图](../../work_records/26-6/260618/group_early_final_heatmap.png)

奖励函数表：

| run | label | epochs | reward | target |
| --- | --- | --- | --- | --- |
| [DG-MAIN-01](../runs/diffgui-8gpu-chembl21-dock_20260609_224313) | Dock+classifier v0.2.1 | 21 | docking 0.5 + chembl_classifier 0.5 | chembl_classifier_reward |
| [DG-MAIN-02](../runs/diffgui-8gpu-docking-only-lr1e5-last2_20260610_094719) | Pure docking short | 5 | docking 1 | docking_reward |
| [DG-MAIN-03](../runs/diffgui-8gpu-ring-last3-lr3e5_20260610_114650) | Ring reward last3 | 23 | docking 0.35 + ligand_efficiency 0.25 + ring_complexity 0.3 + chemistry_guard 0.1 | ring_complexity_reward |
| [DG-MAIN-04](../runs/diffgui-8gpu-b36-classifier-v0-3-1-ppo4_20260610_221410) | Classifier v0.3.1 short | 8 | docking 0.5 + chembl_classifier 0.5 | chembl_classifier_reward |

| run_link | label | epochs | last_reward | delta_reward_total_mean | target_component_first_mean | target_component_last_mean | delta_target_component_mean | last_valid_ratio | last_docking_success_ratio_total | last_descriptor_mmd_vs_baseline | last_aromatic_ring_mean | last_three_member_ring_mean | last_PPO_ApproxKL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [DG-MAIN-01](../runs/diffgui-8gpu-chembl21-dock_20260609_224313) | Dock+classifier v0.2.1 | 21 | 0.288 | -0.032 | -0.116 | -0.079 | 0.036 | 0.875 | 0.875 | 0.300 | 1.452 | 0.268 | 2.301 |
| [DG-MAIN-02](../runs/diffgui-8gpu-docking-only-lr1e5-last2_20260610_094719) | Pure docking short | 5 | 0.764 | -0.055 | 0.984 | 0.969 | -0.015 | 0.885 | 0.885 | 0.328 | 1.347 | 0.176 | 3.667 |
| [DG-MAIN-03](../runs/diffgui-8gpu-ring-last3-lr3e5_20260610_114650) | Ring reward last3 | 23 | 0.636 | 0.230 | 0.279 | 0.375 | 0.097 | 0.984 | 0.984 | 0.208 | 0.894 | 0.127 | 1.380 |
| [DG-MAIN-04](../runs/diffgui-8gpu-b36-classifier-v0-3-1-ppo4_20260610_221410) | Classifier v0.3.1 short | 8 | 0.010 | 0.071 | -0.791 | -0.662 | 0.130 | 0.861 | 0.861 | 0.327 | 1.453 | 0.130 | 3.222 |

- `Dock+classifier v0.2.1` 是最早完整正式训练，21 个 summary epoch 后 reward_total_mean 首末下降，说明直接迁移 docking + 老 MLP classifier 不够。
- `Pure docking short` 虽只有 5 轮，但作为关键短训显示单独 docking 没有形成明确改善趋势，不能当主策略。
- `Ring reward last3` 是该组最明确的正向训练：reward_total_mean 0.406->0.636，valid 0.833->0.984；但芳香环下降、大环/复杂拓扑仍需新 reward 拆分。
- `Classifier v0.3.1 short` 单 epoch 成本高，8 轮内 reward_total_mean 仍低；最终 PPO_ApproxKL=3.222，需要先做 classifier 校准和更小 PPO 设置。

### Reward 归因实验 / 单奖励矩阵

![Reward 归因实验 / 单奖励矩阵 训练曲线](../../work_records/26-6/260618/group_reward_attribution_training_curves.png)

![Reward 归因实验 / 单奖励矩阵 生成分子性质](../../work_records/26-6/260618/group_reward_attribution_molecule_properties.png)

![Reward 归因实验 / 单奖励矩阵 最终指标热图](../../work_records/26-6/260618/group_reward_attribution_final_heatmap.png)

奖励函数表：

| run | label | epochs | reward | target |
| --- | --- | --- | --- | --- |
| [DG-ATTR-01](../runs/diffgui-ring-control-20260613-01-docking_20260613_153723) | docking | 10 | docking 1 | docking_reward |
| [DG-ATTR-02](../runs/diffgui-ring-control-20260613-02-ligand_efficiency_20260613_181441) | ligand_efficiency | 10 | ligand_efficiency 1 | ligand_efficiency_reward |
| [DG-ATTR-03](../runs/diffgui-ring-control-20260613-05-aromatic_scaffold_20260613_204920) | aromatic_scaffold | 10 | aromatic_scaffold 1 | aromatic_scaffold_reward |
| [DG-ATTR-04](../runs/diffgui-ring-control-20260613-07-ring_pathology_20260613_232423) | ring_pathology | 10 | ring_pathology 1 | ring_pathology_reward |
| [DG-ATTR-05](../runs/diffgui-ring-control-20260613-08-chembl_classifier_20260614_020007) | chembl_classifier | 10 | chembl_classifier 1 | chembl_classifier_reward |
| [DG-ATTR-06](../runs/diffgui-ring-control-20260615-09-qed_20260615_122202) | qed | 10 | qed 1 | qed_reward |
| [DG-ATTR-07](../runs/diffgui-ring-control-20260615-10-sa_20260615_163654) | sa | 10 | sa 1 | sa_reward |
| [DG-ATTR-08](../runs/diffgui-ring-control-20260615-11-ring_complexity_20260615_200439) | ring_complexity | 10 | ring_complexity 1 | ring_complexity_reward |

| run_link | label | epochs_with_summary | last_reward_total_mean | delta_reward_total_mean | target_component_first_mean | target_component_last_mean | delta_target_component_mean | last_valid_ratio | last_docking_success_ratio_total | last_descriptor_mmd_vs_baseline | last_aromatic_ring_mean | last_three_member_ring_mean | last_PPO_ApproxKL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [DG-ATTR-01](../runs/diffgui-ring-control-20260613-01-docking_20260613_153723) | docking | 10 | 0.765 | 0.045 | 0.951 | 0.940 | -0.010 | 0.903 | 0.903 | 0.351 | 1.108 | 0.115 | 4.586 |
| [DG-ATTR-02](../runs/diffgui-ring-control-20260613-02-ligand_efficiency_20260613_181441) | ligand_efficiency | 10 | 0.363 | -0.074 | 0.545 | 0.510 | -0.035 | 0.896 | 0.896 | 0.226 | 1.178 | 0.171 | 1.744 |
| [DG-ATTR-03](../runs/diffgui-ring-control-20260613-05-aromatic_scaffold_20260613_204920) | aromatic_scaffold | 10 | -0.019 | 0.081 | -0.019 | 0.054 | 0.073 | 0.917 | 0.917 | 0.266 | 1.227 | 0.167 | 2.791 |
| [DG-ATTR-04](../runs/diffgui-ring-control-20260613-07-ring_pathology_20260613_232423) | ring_pathology | 10 | 0.157 | 0.058 | 0.297 | 0.301 | 0.004 | 0.875 | 0.875 | 0.292 | 1.310 | 0.167 | 2.880 |
| [DG-ATTR-05](../runs/diffgui-ring-control-20260613-08-chembl_classifier_20260614_020007) | chembl_classifier | 10 | -0.619 | 0.197 | -0.792 | -0.571 | 0.221 | 0.882 | 0.882 | 0.282 | 1.323 | 0.157 | 2.166 |
| [DG-ATTR-06](../runs/diffgui-ring-control-20260615-09-qed_20260615_122202) | qed | 10 | 0.390 | -0.089 | 0.626 | 0.614 | -0.012 | 0.861 | 0.861 | 0.297 | 1.089 | 0.258 | 2.836 |
| [DG-ATTR-07](../runs/diffgui-ring-control-20260615-10-sa_20260615_163654) | sa | 10 | 0.469 | -0.056 | 0.663 | 0.640 | -0.024 | 0.889 | 0.889 | 0.240 | 1.148 | 0.203 | 2.393 |
| [DG-ATTR-08](../runs/diffgui-ring-control-20260615-11-ring_complexity_20260615_200439) | ring_complexity | 10 | 0.251 | -0.047 | 0.461 | 0.386 | -0.074 | 0.903 | 0.903 | 0.256 | 1.146 | 0.162 | 2.441 |

- 该组用于判断单个 reward 是否真的推动自身目标组件，而不是只看总分。
- 目标组件首末上升的单奖励包括：aromatic_scaffold, ring_pathology, chembl_classifier。
- 目标组件没有上升或下降的单奖励包括：docking, ligand_efficiency, qed, sa, ring_complexity；这些 reward 单独使用时不稳定，可能受奖励尺度、优势归一化、采样噪声和 PPO 更新强度影响。
- `docking` 最终总分高，但 docking_reward 目标组件略降，必须用独立重打分验证 top 分子。

目标组件首末变化：

| run_alias | label | component | first_mean | last_mean | delta_mean |
| --- | --- | --- | --- | --- | --- |
| DG-ATTR-01 | docking | docking_reward | 0.951 | 0.940 | -0.010 |
| DG-ATTR-02 | ligand_efficiency | ligand_efficiency_reward | 0.545 | 0.510 | -0.035 |
| DG-ATTR-03 | aromatic_scaffold | aromatic_scaffold_reward | -0.019 | 0.054 | 0.073 |
| DG-ATTR-04 | ring_pathology | ring_pathology_reward | 0.297 | 0.301 | 0.004 |
| DG-ATTR-05 | chembl_classifier | chembl_classifier_reward | -0.792 | -0.571 | 0.221 |
| DG-ATTR-06 | qed | qed_reward | 0.626 | 0.614 | -0.012 |
| DG-ATTR-07 | sa | sa_reward | 0.663 | 0.640 | -0.024 |
| DG-ATTR-08 | ring_complexity | ring_complexity_reward | 0.461 | 0.386 | -0.074 |

### 旧组合奖励 / 环控制矩阵

![旧组合奖励 / 环控制矩阵 训练曲线](../../work_records/26-6/260618/group_combination_training_curves.png)

![旧组合奖励 / 环控制矩阵 生成分子性质](../../work_records/26-6/260618/group_combination_molecule_properties.png)

![旧组合奖励 / 环控制矩阵 最终指标热图](../../work_records/26-6/260618/group_combination_final_heatmap.png)

奖励函数表：

| run | label | epochs | reward | target |
| --- | --- | --- | --- | --- |
| [DG-COMB-01](../runs/diffgui-ring-control-20260615-12-le_ring_aromatic_20260615_224513) | LE + ring + aromatic | 10 | ligand_efficiency 0.34 + aromatic_scaffold 0.33 + ring_complexity 0.33 |  |
| [DG-COMB-02](../runs/diffgui-ring-control-20260615-13-dock_ring_aromatic_20260616_011801) | docking + ring + aromatic | 10 | docking 0.34 + aromatic_scaffold 0.33 + ring_complexity 0.33 |  |

| run_link | label | epochs_with_summary | last_reward_total_mean | delta_reward_total_mean | target_component_first_mean | target_component_last_mean | delta_target_component_mean | last_valid_ratio | last_docking_success_ratio_total | last_descriptor_mmd_vs_baseline | last_aromatic_ring_mean | last_three_member_ring_mean | last_PPO_ApproxKL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [DG-COMB-01](../runs/diffgui-ring-control-20260615-12-le_ring_aromatic_20260615_224513) | LE + ring + aromatic | 10 | 0.234 | 0.064 |  |  |  | 0.910 | 0.910 | 0.232 | 1.099 | 0.122 | 2.124 |
| [DG-COMB-02](../runs/diffgui-ring-control-20260615-13-dock_ring_aromatic_20260616_011801) | docking + ring + aromatic | 10 | 0.362 | 0.097 |  |  |  | 0.924 | 0.924 | 0.243 | 1.068 | 0.203 | 2.099 |

- 两个组合奖励都是 10 轮矩阵测试，目标是把 docking/LE 与 ring/aromatic 约束组合起来。
- `LE + ring + aromatic` 的 MMD 和三元环更好，适合作为保守分布控制对照。
- `docking + ring + aromatic` 有效率最高，但三元环和 ring_count 仍偏高，说明 docking 加入后会把结构病理压力重新拉回来。

### 核心 next6 / 环病理组合矩阵
把有效率/MMD 换好了，但牺牲了 docking 和芳香结构保持

![核心 next6 / 环病理组合矩阵 训练曲线](../../work_records/26-6/260618/group_next6_training_curves.png)

![核心 next6 / 环病理组合矩阵 生成分子性质](../../work_records/26-6/260618/group_next6_molecule_properties.png)

![核心 next6 / 环病理组合矩阵 最终指标热图](../../work_records/26-6/260618/group_next6_final_heatmap.png)

奖励函数表：

| run | label | epochs | reward | target |
| --- | --- | --- | --- | --- |
| [DG-NEXT-01](../runs/diffgui-next6-20260618-01-dock_aromatic_pathology_20260618_211501) | dock + aromatic + pathology | 20 | docking 0.4 + aromatic_scaffold 0.3 + ring_pathology 0.3 |  |
| [DG-NEXT-02](../runs/diffgui-next6-20260618-02-le_aromatic_pathology_20260619_020218) | LE + aromatic + pathology | 20 | ligand_efficiency 0.4 + aromatic_scaffold 0.3 + ring_pathology 0.3 |  |
| [DG-NEXT-03](../runs/diffgui-next6-20260618-03-dock_le_aromatic_pathology_20260619_064544) | dock + LE + aromatic + pathology | 20 | docking 0.3 + ligand_efficiency 0.25 + aromatic_scaffold 0.25 + ring_pathology 0.2 |  |
| [DG-NEXT-04](../runs/diffgui-next6-20260618-04-ring_last3_pathology_guard_20260619_112410) | ring last3 + pathology + guard | 20 | docking 0.3 + ligand_efficiency 0.2 + aromatic_scaffold 0.25 + ring_pathology 0.15 + chemistry_guard 0.1 |  |
| [DG-NEXT-05](../runs/diffgui-next6-20260618-05-chembl_aromatic_pathology_20260619_160546) | chembl + aromatic + pathology | 20 | aromatic_scaffold 0.3 + ring_pathology 0.3 + chembl_classifier 0.4 |  |
| [DG-NEXT-06](../runs/diffgui-next6-20260618-06-dock_chembl_aromatic_pathology_20260619_204502) | dock + chembl + aromatic + pathology | 20 | docking 0.3 + aromatic_scaffold 0.25 + ring_pathology 0.2 + chembl_classifier 0.25 |  |

| run_link | label | epochs_with_summary | last_reward_total_mean | delta_reward_total_mean | target_component_first_mean | target_component_last_mean | delta_target_component_mean | last_valid_ratio | last_docking_success_ratio_total | last_descriptor_mmd_vs_baseline | last_aromatic_ring_mean | last_three_member_ring_mean | last_PPO_ApproxKL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [DG-NEXT-01](../runs/diffgui-next6-20260618-01-dock_aromatic_pathology_20260618_211501) | dock + aromatic + pathology | 20 | 0.418 | 0.119 |  |  |  | 0.958 | 0.965 | 0.236 | 0.928 | 0.152 | 2.022 |
| [DG-NEXT-02](../runs/diffgui-next6-20260618-02-le_aromatic_pathology_20260619_020218) | LE + aromatic + pathology | 20 | 0.216 | 0.050 |  |  |  | 0.986 | 0.986 | 0.183 | 0.775 | 0.183 | 2.268 |
| [DG-NEXT-03](../runs/diffgui-next6-20260618-03-dock_le_aromatic_pathology_20260619_064544) | dock + LE + aromatic + pathology | 20 | 0.417 | 0.131 |  |  |  | 0.944 | 0.944 | 0.218 | 0.860 | 0.213 | 2.026 |
| [DG-NEXT-04](../runs/diffgui-next6-20260618-04-ring_last3_pathology_guard_20260619_112410) | ring last3 + pathology + guard | 20 | 0.457 | 0.109 |  |  |  | 0.951 | 0.958 | 0.213 | 0.985 | 0.168 | 1.935 |
| [DG-NEXT-05](../runs/diffgui-next6-20260618-05-chembl_aromatic_pathology_20260619_160546) | chembl + aromatic + pathology | 20 | -0.234 | 0.082 |  |  |  | 0.917 | 0.917 | 0.214 | 0.841 | 0.129 | 2.496 |
| [DG-NEXT-06](../runs/diffgui-next6-20260618-06-dock_chembl_aromatic_pathology_20260619_204502) | dock + chembl + aromatic + pathology | 20 | 0.022 | -0.029 |  |  |  | 0.910 | 0.910 | 0.191 | 0.664 | 0.145 | 3.673 |

- 这 6 个 run 是当前 DiffGui 总结的核心证据：同为 8GPU、20 epoch、batch/rank=18 的组合矩阵，重点比较 aromatic_scaffold、ring_pathology、docking、LE、chembl_classifier 和 chemistry_guard 的取舍。
- `DG-NEXT-02 LE + aromatic + pathology` 同时给出最高 valid=0.986 和最低 MMD=0.183，适合作为当前最稳的分布/合法性基线。
- `DG-NEXT-04 ring last3 + pathology + guard` 的 aromatic_ring_mean=0.985 最高，说明 ring-last3 思路加 pathology/guard 后更能保住芳香骨架。
- `DG-NEXT-05 chembl + aromatic + pathology` 的 three_member_ring_mean=0.129 最低，更适合作为压结构病理的对照，但 reward_total_mean 为负，不能按总 reward 选它。
- 加 docking 后 docking success 维持高位，但不会自动改善 MMD、芳香或三元环；后续组合需要按结构安全和分布指标一起筛。

## Reward 归因组件首末变化

| group | run_alias | label | component | first_mean | last_mean | delta_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 早期正式大训练 / 背景短训 | DG-MAIN-01 | Dock+classifier v0.2.1 | chembl_classifier_reward | -0.116 | -0.079 | 0.036 |
| 早期正式大训练 / 背景短训 | DG-MAIN-02 | Pure docking short | docking_reward | 0.984 | 0.969 | -0.015 |
| 早期正式大训练 / 背景短训 | DG-MAIN-03 | Ring reward last3 | ring_complexity_reward | 0.279 | 0.375 | 0.097 |
| 早期正式大训练 / 背景短训 | DG-MAIN-04 | Classifier v0.3.1 short | chembl_classifier_reward | -0.791 | -0.662 | 0.130 |
| Reward 归因实验 / 单奖励矩阵 | DG-ATTR-01 | docking | docking_reward | 0.951 | 0.940 | -0.010 |
| Reward 归因实验 / 单奖励矩阵 | DG-ATTR-02 | ligand_efficiency | ligand_efficiency_reward | 0.545 | 0.510 | -0.035 |
| Reward 归因实验 / 单奖励矩阵 | DG-ATTR-03 | aromatic_scaffold | aromatic_scaffold_reward | -0.019 | 0.054 | 0.073 |
| Reward 归因实验 / 单奖励矩阵 | DG-ATTR-04 | ring_pathology | ring_pathology_reward | 0.297 | 0.301 | 0.004 |
| Reward 归因实验 / 单奖励矩阵 | DG-ATTR-05 | chembl_classifier | chembl_classifier_reward | -0.792 | -0.571 | 0.221 |
| Reward 归因实验 / 单奖励矩阵 | DG-ATTR-06 | qed | qed_reward | 0.626 | 0.614 | -0.012 |
| Reward 归因实验 / 单奖励矩阵 | DG-ATTR-07 | sa | sa_reward | 0.663 | 0.640 | -0.024 |
| Reward 归因实验 / 单奖励矩阵 | DG-ATTR-08 | ring_complexity | ring_complexity_reward | 0.461 | 0.386 | -0.074 |

## 当前判断与下一步

- DiffGui PPO 工程链路已经能支撑 8GPU、200 step、20 epoch 的 next6 串行矩阵；时间瓶颈仍主要在 PPO 更新。
- next6 中 valid 与 MMD 最优均为 `DG-NEXT-02 LE + aromatic + pathology`：valid=0.986，MMD=0.183，这是当前最稳的分布/合法性方案。
- aromatic 保留最好的是 `DG-NEXT-04 ring last3 + pathology + guard`，aromatic_ring_mean=0.985；这是旧 ring-last3 思路加 pathology/guard 后的主要价值。
- 三元环最低的是 `DG-NEXT-05 chembl + aromatic + pathology`，three_member_ring_mean=0.129，说明 chembl + aromatic + pathology 更像结构病理抑制对照。
- 加 docking 不是单调变好：`dock + aromatic + pathology` docking success 高，但 MMD 不如 LE 方案；`dock + chembl + aromatic + pathology` 的 aromatic 明显偏低。
- 早期 `Ring reward last3` 仍作为背景基线保留：valid=0.984，MMD=0.208，但 aromatic_ring_mean 下降，所以现在应以 next6 的 aromatic/pathology 拆分矩阵作为核心证据。
- Reward 归因实验继续定位为诊断工具：`aromatic_scaffold`、`chembl_classifier`、`ring_pathology` 的目标组件有正向信号；`QED/SA/LE/ring_complexity` 单独优化不稳定。
- 下一步应围绕 `DG-NEXT-02`、`DG-NEXT-04`、`DG-NEXT-05` 做等量离线采样、独立 docking/D-MPNN 重打分和结构审计；不要只按 `reward_total_mean` 选。

## 未纳入主要表的 DiffGui run

| run_dir | config_name | epochs_with_summary | reason |
| --- | --- | --- | --- |
| runs/diffgui-8gpu-aromatic-last3-lr3e5_20260610_195621 | diffgui-8gpu-b24-aromatic-last3-lr3e5 | 0 | no training_summary.csv or no complete summary epoch |
| runs/diffgui-8gpu-chembl21-dock_20260609_214541 | diffgui-8gpu-b24-chembl21-dock | 0 | no training_summary.csv or no complete summary epoch |
| runs/diffgui-8gpu-chembl21-dock_20260609_214844 | diffgui-8gpu-b24-chembl21-dock | 0 | no training_summary.csv or no complete summary epoch |
| runs/diffgui-8gpu-chembl21-dock_20260609_215106 | diffgui-8gpu-b24-chembl21-dock | 0 | no training_summary.csv or no complete summary epoch |
| runs/diffgui-8gpu-chembl21-dock_20260609_221306 | diffgui-8gpu-b24-chembl21-dock | 0 | no training_summary.csv or no complete summary epoch |
| runs/diffgui-docking-ddp-smoke_20260609_204340 |  | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-docking-ddp-smoke_20260609_204434 | diffgui-docking-ddp-smoke | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-docking-ddp-smoke_20260609_204811 | diffgui-docking-ddp-smoke | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-docking-ddp-smoke_20260609_211432 |  | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-docking-ddp-smoke_20260609_213459 |  | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-docking-ddp-smoke_20260609_213609 | diffgui-docking-ddp-smoke | 1 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-docking-stage1-last1-b16_test_20260609 | diffgui-docking-stage1-last1-b16 | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-docking-train-smoke_20260609_atomsize_predfix | diffgui-docking-train-smoke | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-docking-train-smoke_20260609_codex | diffgui-docking-train-smoke | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-replay-gate-gpu1-smoke_20260616_1728 | diffgui-replay-gate-gpu1-smoke | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-replay-gate-gpu1-smoke_20260616_1740 | diffgui-replay-gate-gpu1-smoke | 0 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-replay-gate-gpu1-smoke_20260616_174444 | diffgui-replay-gate-gpu1-smoke | 2 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-replay-gate-gpu1-smoke_20260616_193444 | diffgui-replay-gate-gpu1-smoke | 2 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-replay-gate-gpu1-smoke_20260616_194226 | diffgui-replay-gate-gpu1-smoke | 2 | smoke/debug/replay-gate/preflight run |
| runs/diffgui-ring-control-20260615-09-qed_20260615_122001 | diffgui-ring-control-20260615-09-qed | 0 | no training_summary.csv or no complete summary epoch |

## 附表：主要训练总览

| group | run_link | label | epochs_with_summary | per_rank_batch_size | global_batch_size | num_timesteps | lr | ppo_epochs | freeze_mode | train_last_n_blocks | last_reward_total_mean | delta_reward_total_mean | last_valid_ratio | last_docking_success_ratio_total | last_descriptor_mmd_vs_baseline | last_ring_count_mean | last_aromatic_ring_mean | last_three_member_ring_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 早期正式大训练 / 背景短训 | [DG-MAIN-01](../runs/diffgui-8gpu-chembl21-dock_20260609_224313) | Dock+classifier v0.2.1 | 21 | 24 | 192 | 200 | 1e-06 | 2 | last_n_blocks | 1 | 0.288 | -0.032 | 0.875 | 0.875 | 0.300 | 4.238 | 1.452 | 0.268 |
| 早期正式大训练 / 背景短训 | [DG-MAIN-02](../runs/diffgui-8gpu-docking-only-lr1e5-last2_20260610_094719) | Pure docking short | 5 | 24 | 192 | 200 | 1e-05 | 1 | last_n_blocks | 2 | 0.764 | -0.055 | 0.885 | 0.885 | 0.328 | 3.976 | 1.347 | 0.176 |
| 早期正式大训练 / 背景短训 | [DG-MAIN-03](../runs/diffgui-8gpu-ring-last3-lr3e5_20260610_114650) | Ring reward last3 | 23 | 24 | 192 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.636 | 0.230 | 0.984 | 0.984 | 0.208 | 4.143 | 0.894 | 0.127 |
| 早期正式大训练 / 背景短训 | [DG-MAIN-04](../runs/diffgui-8gpu-b36-classifier-v0-3-1-ppo4_20260610_221410) | Classifier v0.3.1 short | 8 | 36 | 288 | 200 | 1e-06 | 4 | last_n_blocks | 1 | 0.010 | 0.071 | 0.861 | 0.861 | 0.327 | 3.947 | 1.453 | 0.130 |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-01](../runs/diffgui-ring-control-20260613-01-docking_20260613_153723) | docking | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.765 | 0.045 | 0.903 | 0.903 | 0.351 | 3.892 | 1.108 | 0.115 |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-02](../runs/diffgui-ring-control-20260613-02-ligand_efficiency_20260613_181441) | ligand_efficiency | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.363 | -0.074 | 0.896 | 0.896 | 0.226 | 4.093 | 1.178 | 0.171 |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-03](../runs/diffgui-ring-control-20260613-05-aromatic_scaffold_20260613_204920) | aromatic_scaffold | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | -0.019 | 0.081 | 0.917 | 0.917 | 0.266 | 4.008 | 1.227 | 0.167 |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-04](../runs/diffgui-ring-control-20260613-07-ring_pathology_20260613_232423) | ring_pathology | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.157 | 0.058 | 0.875 | 0.875 | 0.292 | 3.841 | 1.310 | 0.167 |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-05](../runs/diffgui-ring-control-20260613-08-chembl_classifier_20260614_020007) | chembl_classifier | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | -0.619 | 0.197 | 0.882 | 0.882 | 0.282 | 3.701 | 1.323 | 0.157 |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-06](../runs/diffgui-ring-control-20260615-09-qed_20260615_122202) | qed | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.390 | -0.089 | 0.861 | 0.861 | 0.297 | 4.024 | 1.089 | 0.258 |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-07](../runs/diffgui-ring-control-20260615-10-sa_20260615_163654) | sa | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.469 | -0.056 | 0.889 | 0.889 | 0.240 | 3.789 | 1.148 | 0.203 |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-08](../runs/diffgui-ring-control-20260615-11-ring_complexity_20260615_200439) | ring_complexity | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.251 | -0.047 | 0.903 | 0.903 | 0.256 | 4.092 | 1.146 | 0.162 |
| 旧组合奖励 / 环控制矩阵 | [DG-COMB-01](../runs/diffgui-ring-control-20260615-12-le_ring_aromatic_20260615_224513) | LE + ring + aromatic | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.234 | 0.064 | 0.910 | 0.910 | 0.232 | 3.702 | 1.099 | 0.122 |
| 旧组合奖励 / 环控制矩阵 | [DG-COMB-02](../runs/diffgui-ring-control-20260615-13-dock_ring_aromatic_20260616_011801) | docking + ring + aromatic | 10 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.362 | 0.097 | 0.924 | 0.924 | 0.243 | 4.120 | 1.068 | 0.203 |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-01](../runs/diffgui-next6-20260618-01-dock_aromatic_pathology_20260618_211501) | dock + aromatic + pathology | 20 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.418 | 0.119 | 0.958 | 0.965 | 0.236 | 3.906 | 0.928 | 0.152 |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-02](../runs/diffgui-next6-20260618-02-le_aromatic_pathology_20260619_020218) | LE + aromatic + pathology | 20 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.216 | 0.050 | 0.986 | 0.986 | 0.183 | 4.007 | 0.775 | 0.183 |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-03](../runs/diffgui-next6-20260618-03-dock_le_aromatic_pathology_20260619_064544) | dock + LE + aromatic + pathology | 20 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.417 | 0.131 | 0.944 | 0.944 | 0.218 | 3.809 | 0.860 | 0.213 |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-04](../runs/diffgui-next6-20260618-04-ring_last3_pathology_guard_20260619_112410) | ring last3 + pathology + guard | 20 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.457 | 0.109 | 0.951 | 0.958 | 0.213 | 4.131 | 0.985 | 0.168 |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-05](../runs/diffgui-next6-20260618-05-chembl_aromatic_pathology_20260619_160546) | chembl + aromatic + pathology | 20 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | -0.234 | 0.082 | 0.917 | 0.917 | 0.214 | 3.742 | 0.841 | 0.129 |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-06](../runs/diffgui-next6-20260618-06-dock_chembl_aromatic_pathology_20260619_204502) | dock + chembl + aromatic + pathology | 20 | 18 | 144 | 200 | 3e-05 | 1 | last_n_blocks | 3 | 0.022 | -0.029 | 0.910 | 0.910 | 0.191 | 3.748 | 0.664 | 0.145 |

## 附表：奖励函数总表

下表不按研究线分组，单独列出每个主要 run 的 reward 组成和归因目标，便于横向检查配置差异。

| group | run | label | epochs | reward | target |
| --- | --- | --- | --- | --- | --- |
| 早期正式大训练 / 背景短训 | [DG-MAIN-01](../runs/diffgui-8gpu-chembl21-dock_20260609_224313) | Dock+classifier v0.2.1 | 21 | docking 0.5 + chembl_classifier 0.5 | chembl_classifier_reward |
| 早期正式大训练 / 背景短训 | [DG-MAIN-02](../runs/diffgui-8gpu-docking-only-lr1e5-last2_20260610_094719) | Pure docking short | 5 | docking 1 | docking_reward |
| 早期正式大训练 / 背景短训 | [DG-MAIN-03](../runs/diffgui-8gpu-ring-last3-lr3e5_20260610_114650) | Ring reward last3 | 23 | docking 0.35 + ligand_efficiency 0.25 + ring_complexity 0.3 + chemistry_guard 0.1 | ring_complexity_reward |
| 早期正式大训练 / 背景短训 | [DG-MAIN-04](../runs/diffgui-8gpu-b36-classifier-v0-3-1-ppo4_20260610_221410) | Classifier v0.3.1 short | 8 | docking 0.5 + chembl_classifier 0.5 | chembl_classifier_reward |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-01](../runs/diffgui-ring-control-20260613-01-docking_20260613_153723) | docking | 10 | docking 1 | docking_reward |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-02](../runs/diffgui-ring-control-20260613-02-ligand_efficiency_20260613_181441) | ligand_efficiency | 10 | ligand_efficiency 1 | ligand_efficiency_reward |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-03](../runs/diffgui-ring-control-20260613-05-aromatic_scaffold_20260613_204920) | aromatic_scaffold | 10 | aromatic_scaffold 1 | aromatic_scaffold_reward |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-04](../runs/diffgui-ring-control-20260613-07-ring_pathology_20260613_232423) | ring_pathology | 10 | ring_pathology 1 | ring_pathology_reward |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-05](../runs/diffgui-ring-control-20260613-08-chembl_classifier_20260614_020007) | chembl_classifier | 10 | chembl_classifier 1 | chembl_classifier_reward |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-06](../runs/diffgui-ring-control-20260615-09-qed_20260615_122202) | qed | 10 | qed 1 | qed_reward |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-07](../runs/diffgui-ring-control-20260615-10-sa_20260615_163654) | sa | 10 | sa 1 | sa_reward |
| Reward 归因实验 / 单奖励矩阵 | [DG-ATTR-08](../runs/diffgui-ring-control-20260615-11-ring_complexity_20260615_200439) | ring_complexity | 10 | ring_complexity 1 | ring_complexity_reward |
| 旧组合奖励 / 环控制矩阵 | [DG-COMB-01](../runs/diffgui-ring-control-20260615-12-le_ring_aromatic_20260615_224513) | LE + ring + aromatic | 10 | ligand_efficiency 0.34 + aromatic_scaffold 0.33 + ring_complexity 0.33 |  |
| 旧组合奖励 / 环控制矩阵 | [DG-COMB-02](../runs/diffgui-ring-control-20260615-13-dock_ring_aromatic_20260616_011801) | docking + ring + aromatic | 10 | docking 0.34 + aromatic_scaffold 0.33 + ring_complexity 0.33 |  |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-01](../runs/diffgui-next6-20260618-01-dock_aromatic_pathology_20260618_211501) | dock + aromatic + pathology | 20 | docking 0.4 + aromatic_scaffold 0.3 + ring_pathology 0.3 |  |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-02](../runs/diffgui-next6-20260618-02-le_aromatic_pathology_20260619_020218) | LE + aromatic + pathology | 20 | ligand_efficiency 0.4 + aromatic_scaffold 0.3 + ring_pathology 0.3 |  |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-03](../runs/diffgui-next6-20260618-03-dock_le_aromatic_pathology_20260619_064544) | dock + LE + aromatic + pathology | 20 | docking 0.3 + ligand_efficiency 0.25 + aromatic_scaffold 0.25 + ring_pathology 0.2 |  |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-04](../runs/diffgui-next6-20260618-04-ring_last3_pathology_guard_20260619_112410) | ring last3 + pathology + guard | 20 | docking 0.3 + ligand_efficiency 0.2 + aromatic_scaffold 0.25 + ring_pathology 0.15 + chemistry_guard 0.1 |  |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-05](../runs/diffgui-next6-20260618-05-chembl_aromatic_pathology_20260619_160546) | chembl + aromatic + pathology | 20 | aromatic_scaffold 0.3 + ring_pathology 0.3 + chembl_classifier 0.4 |  |
| 核心 next6 / 环病理组合矩阵 | [DG-NEXT-06](../runs/diffgui-next6-20260618-06-dock_chembl_aromatic_pathology_20260619_204502) | dock + chembl + aromatic + pathology | 20 | docking 0.3 + aromatic_scaffold 0.25 + ring_pathology 0.2 + chembl_classifier 0.25 |  |

## 附表：别名映射表

| run_alias | label | group | run_dir |
| --- | --- | --- | --- |
| DG-MAIN-01 | Dock+classifier v0.2.1 | 早期正式大训练 / 背景短训 | runs/diffgui-8gpu-chembl21-dock_20260609_224313 |
| DG-MAIN-02 | Pure docking short | 早期正式大训练 / 背景短训 | runs/diffgui-8gpu-docking-only-lr1e5-last2_20260610_094719 |
| DG-MAIN-03 | Ring reward last3 | 早期正式大训练 / 背景短训 | runs/diffgui-8gpu-ring-last3-lr3e5_20260610_114650 |
| DG-MAIN-04 | Classifier v0.3.1 short | 早期正式大训练 / 背景短训 | runs/diffgui-8gpu-b36-classifier-v0-3-1-ppo4_20260610_221410 |
| DG-ATTR-01 | docking | Reward 归因实验 / 单奖励矩阵 | runs/diffgui-ring-control-20260613-01-docking_20260613_153723 |
| DG-ATTR-02 | ligand_efficiency | Reward 归因实验 / 单奖励矩阵 | runs/diffgui-ring-control-20260613-02-ligand_efficiency_20260613_181441 |
| DG-ATTR-03 | aromatic_scaffold | Reward 归因实验 / 单奖励矩阵 | runs/diffgui-ring-control-20260613-05-aromatic_scaffold_20260613_204920 |
| DG-ATTR-04 | ring_pathology | Reward 归因实验 / 单奖励矩阵 | runs/diffgui-ring-control-20260613-07-ring_pathology_20260613_232423 |
| DG-ATTR-05 | chembl_classifier | Reward 归因实验 / 单奖励矩阵 | runs/diffgui-ring-control-20260613-08-chembl_classifier_20260614_020007 |
| DG-ATTR-06 | qed | Reward 归因实验 / 单奖励矩阵 | runs/diffgui-ring-control-20260615-09-qed_20260615_122202 |
| DG-ATTR-07 | sa | Reward 归因实验 / 单奖励矩阵 | runs/diffgui-ring-control-20260615-10-sa_20260615_163654 |
| DG-ATTR-08 | ring_complexity | Reward 归因实验 / 单奖励矩阵 | runs/diffgui-ring-control-20260615-11-ring_complexity_20260615_200439 |
| DG-COMB-01 | LE + ring + aromatic | 旧组合奖励 / 环控制矩阵 | runs/diffgui-ring-control-20260615-12-le_ring_aromatic_20260615_224513 |
| DG-COMB-02 | docking + ring + aromatic | 旧组合奖励 / 环控制矩阵 | runs/diffgui-ring-control-20260615-13-dock_ring_aromatic_20260616_011801 |
| DG-NEXT-01 | dock + aromatic + pathology | 核心 next6 / 环病理组合矩阵 | runs/diffgui-next6-20260618-01-dock_aromatic_pathology_20260618_211501 |
| DG-NEXT-02 | LE + aromatic + pathology | 核心 next6 / 环病理组合矩阵 | runs/diffgui-next6-20260618-02-le_aromatic_pathology_20260619_020218 |
| DG-NEXT-03 | dock + LE + aromatic + pathology | 核心 next6 / 环病理组合矩阵 | runs/diffgui-next6-20260618-03-dock_le_aromatic_pathology_20260619_064544 |
| DG-NEXT-04 | ring last3 + pathology + guard | 核心 next6 / 环病理组合矩阵 | runs/diffgui-next6-20260618-04-ring_last3_pathology_guard_20260619_112410 |
| DG-NEXT-05 | chembl + aromatic + pathology | 核心 next6 / 环病理组合矩阵 | runs/diffgui-next6-20260618-05-chembl_aromatic_pathology_20260619_160546 |
| DG-NEXT-06 | dock + chembl + aromatic + pathology | 核心 next6 / 环病理组合矩阵 | runs/diffgui-next6-20260618-06-dock_chembl_aromatic_pathology_20260619_204502 |

## 输出文件

- [major_runs_summary.csv](../analysis/diffgui_rl_major_runs_20260618/major_runs_summary.csv)
- [major_runs_epoch_metrics.csv](../analysis/diffgui_rl_major_runs_20260618/major_runs_epoch_metrics.csv)
- [reward_component_first_last.csv](../analysis/diffgui_rl_major_runs_20260618/reward_component_first_last.csv)
- [excluded_or_nonmajor_diffgui_runs.csv](../analysis/diffgui_rl_major_runs_20260618/excluded_or_nonmajor_diffgui_runs.csv)
- `../../work_records/26-6/260618/major_runs_key_curve_panels.png`：代表性关键曲线。
- `../../work_records/26-6/260618/major_runs_final_metric_heatmap.png`：全部主要 run 的最终指标热图。
- `../../work_records/26-6/260618/next6_tradeoff_summary.png/svg`：next6 六组核心矩阵的 endpoint tradeoff 图。
- `group_*_training_curves.png`：每条研究线的训练曲线。
- `group_*_molecule_properties.png`：每条研究线的生成分子性质曲线。
- `group_*_final_heatmap.png`：每条研究线的最终指标热图。

## 验证

- `conda run -n dif python -m py_compile analysis/diffgui_rl_major_runs_20260618/build_report.py`：通过。
- `conda run -n dif python analysis/diffgui_rl_major_runs_20260618/build_report.py`：通过，已重建 CSV/JSON、图件和本工作记录。

## 解释限制

这些指标主要来自训练期 rollout 样本和在线 reward。它们适合比较训练轨迹和 reward 响应，但不能替代 checkpoint 离线等量采样、独立 docking/D-MPNN 重打分和人工结构审计。`ring_last3_lr3e5` 的 epoch 24 只有 rank-level 文件，本记录按完整 `training_summary.csv` 的 23 轮解释。
