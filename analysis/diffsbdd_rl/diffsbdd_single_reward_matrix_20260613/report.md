# DiffSBDD 单奖励矩阵九组训练深度分析

## 1. 实验目的

本次九组训练的核心目的不是直接寻找最终最优 reward，而是做单因素诊断：固定 DiffSBDD 强化学习参数，每次只开启一个奖励函数，观察它在 10 个 epoch 内会把分子性质、有效率、docking、分子大小、环结构和分布漂移推向什么方向。

统一训练设置：200 步扩散，8 GPU，每卡 18 分子，全局 batch 144，PPO 复用 2 次，10 epoch，学习率 2e-5，KL 系数 0.05，`kl_mode=analytic_current_ref`，GoodMoleculePool 与 ChemistryGuard 关闭。

## 2. 输出文件

- 长表：`analysis/diffsbdd_single_reward_matrix_20260613/merged_epoch_metrics.csv`
- 终点比较：`analysis/diffsbdd_single_reward_matrix_20260613/final_epoch_comparison.csv`
- 变化量比较：`analysis/diffsbdd_single_reward_matrix_20260613/epoch10_minus_epoch1_delta.csv`
- 训练曲线总览：`analysis/diffsbdd_single_reward_matrix_20260613/training_curves_overview.png`
- 分子性质趋势：`analysis/diffsbdd_single_reward_matrix_20260613/molecular_property_trends.png`
- 环与 scaffold 趋势：`analysis/diffsbdd_single_reward_matrix_20260613/ring_and_scaffold_trends.png`
- 终点柱状对比：`analysis/diffsbdd_single_reward_matrix_20260613/final_epoch_bar_comparison.png`
- epoch10-epoch1 热图：`analysis/diffsbdd_single_reward_matrix_20260613/epoch_delta_heatmap.png`
- 统一离线 docking 每分子结果：`analysis/diffsbdd_single_reward_matrix_20260613/offline_docking_all_epochs_gpu/offline_docking_per_molecule.csv`
- 统一离线 docking 每 epoch 汇总：`analysis/diffsbdd_single_reward_matrix_20260613/offline_docking_all_epochs_gpu/offline_docking_by_epoch.csv`
- 统一离线 docking 趋势图：`analysis/diffsbdd_single_reward_matrix_20260613/offline_docking_all_epochs_gpu/offline_docking_trends.png`

![training curves](../../../analysis/diffsbdd_single_reward_matrix_20260613/training_curves_overview.png)

![property trends](../../../analysis/diffsbdd_single_reward_matrix_20260613/molecular_property_trends.png)

![ring trends](../../../analysis/diffsbdd_single_reward_matrix_20260613/ring_and_scaffold_trends.png)

![final comparison](../../../analysis/diffsbdd_single_reward_matrix_20260613/final_epoch_bar_comparison.png)

![delta heatmap](../../../analysis/diffsbdd_single_reward_matrix_20260613/epoch_delta_heatmap.png)

## 3. 终点总表

| reward            | valid | fragment | dock   | LE    | logP   | TPSA    | C_frac | hetero | heavy  | rings | benzene | pathology | MMD   |
| ----------------- | ----- | -------- | ------ | ----- | ------ | ------- | ------ | ------ | ------ | ----- | ------- | --------- | ----- |
| docking           | 0.611 | 0.333    | -7.619 | 0.363 | 0.263  | 107.97  | 0.641  | 7.883  | 22.114 | 2.352 | 0.034   | 0.42      | 0.219 |
| qed               | 0.597 | 0.319    | -7.73  | 0.366 | 0.194  | 108.658 | 0.641  | 7.992  | 22.547 | 2.349 | 0.047   | 0.442     | 0.213 |
| sa                | 0.562 | 0.347    | -7.856 | 0.367 | 0.29   | 105.74  | 0.65   | 7.784  | 22.58  | 2.556 | 0.062   | 0.556     | 0.204 |
| tpsa_distribution | 0.653 | 0.25     | -7.911 | 0.363 | 0.282  | 109.743 | 0.645  | 7.921  | 23.032 | 2.564 | 0.043   | 0.532     | 0.238 |
| ecfp4_prior       | 0.618 | 0.306    | -7.708 | 0.36  | 0.241  | 110.407 | 0.639  | 7.968  | 22.517 | 2.483 | 0.034   | 0.494     | 0.256 |
| fcfp4_prior       | 0.646 | 0.264    | -7.839 | 0.354 | 0.329  | 113.208 | 0.643  | 8.159  | 23.387 | 2.441 | 0.032   | 0.43      | 0.247 |
| logp_distribution | 0.66  | 0.229    | -7.774 | 0.356 | 0.423  | 106.986 | 0.648  | 7.829  | 23.137 | 2.589 | 0.032   | 0.474     | 0.206 |
| ligand_efficiency | 0.688 | 0.208    | -7.768 | 0.349 | -0.085 | 117.058 | 0.63   | 8.515  | 23.293 | 2.657 | 0.04    | 0.465     | 0.298 |
| chembl_classifier | 0.632 | 0.285    | -7.605 | 0.366 | 0.462  | 102.297 | 0.65   | 7.486  | 22.143 | 2.604 | 0.022   | 0.473     | 0.169 |

## 4. epoch 10 相对 epoch 1 的变化

| reward            | valid_d | dock_d | logP_d | TPSA_d | C_frac_d | heavy_d | pathology_d | MMD_d  |
| ----------------- | ------- | ------ | ------ | ------ | -------- | ------- | ----------- | ------ |
| docking           | 0.0     | -0.043 | -0.107 | 3.072  | -0.008   | -0.83   | -0.034      | -0.001 |
| qed               | -0.062  | -0.087 | -0.231 | 4.236  | -0.017   | -0.759  | -0.053      | -0.016 |
| sa                | -0.069  | -0.182 | -0.186 | 1.876  | -0.012   | -0.552  | 0.061       | -0.001 |
| tpsa_distribution | -0.028  | -0.221 | -0.151 | 4.573  | -0.014   | -0.305  | 0.042       | 0.046  |
| ecfp4_prior       | -0.069  | 0.091  | -0.381 | 4.037  | -0.031   | -1.352  | -0.021      | 0.053  |
| fcfp4_prior       | -0.056  | -0.023 | -0.386 | 4.747  | -0.024   | -0.92   | -0.085      | 0.018  |
| logp_distribution | -0.035  | 0.077  | -0.245 | -0.248 | -0.022   | -0.963  | -0.026      | -0.023 |
| ligand_efficiency | 0.042   | -0.05  | -0.679 | 11.825 | -0.037   | -0.137  | -0.041      | 0.093  |
| chembl_classifier | -0.049  | 0.224  | -0.079 | 0.989  | -0.024   | -0.612  | -0.027      | -0.01  |

## 5. 关键结论

- 有效率最高的是 `ligand_efficiency`，epoch10 RDKit valid ratio=0.688；最低的是 `sa`，valid ratio=0.562。
- 训练时只有 `docking` 和 `ligand_efficiency` 两组启用了 docking provider；本报告已额外对 90 个 epoch SDF 做统一离线 UniDock，因此终点表里的 docking/LE 对九组都可横向比较。
- 统一离线 docking 终点最强的是 `tpsa_distribution`，avg_docking=-7.911；但需要同时看 fragment、carbon、ring pathology，避免只优化对接分数。
- 环病态比例最低的是 `docking`，pathology fraction=0.420；最高的是 `sa`，pathology fraction=0.556。
- 与 ChEMBL 6D 描述符分布最接近的是 `chembl_classifier`，descriptor MMD=0.169。这类指标比单一 reward mean 更能反映分子性质是否整体跑偏。
- 本轮没有观察到高碳漂移：9 组 carbon_fraction 从 epoch1 到 epoch10 均未上升；变化最小的是 `docking` (-0.008)，下降最多的是 `ligand_efficiency` (-0.037)。
- TPSA 整体没有坍缩：除 `logp_distribution` 基本持平 (-0.25) 外，多数组上升；上升最多的是 `ligand_efficiency` (+11.82)。

## 6. 分组分析

### docking

- 终点：valid=0.611，fragment=0.333，dock=-7.619，LE=0.363。
- 分子性质终点：logP=0.263，TPSA=107.970，carbon=0.641，hetero=7.883，heavy=22.114。
- 环结构终点：rings=2.352，benzene=0.034，max fused=1.580，pathology=0.420。
- 训练变化：valid +0.000，logP -0.107，TPSA +3.072，carbon -0.008，pathology -0.034。
- 解读：docking 单独训练最能回答性能上限，但它只关心对接分数，容易忽视分子合法性、极性和结构复杂度。若 docking 改善同时 carbon/pathology 上升，就说明它在利用疏水或复杂结构 shortcut。

### qed

- 终点：valid=0.597，fragment=0.319，dock=-7.730，LE=0.366。
- 分子性质终点：logP=0.194，TPSA=108.658，carbon=0.641，hetero=7.992，heavy=22.547。
- 环结构终点：rings=2.349，benzene=0.047，max fused=1.535，pathology=0.442。
- 训练变化：valid -0.062，logP -0.231，TPSA +4.236，carbon -0.017，pathology -0.053。
- 解读：QED 是通用 drug-like 压力，通常会倾向中等大小、较温和的极性和疏水性。它适合作为稳定化项，但不直接保证靶点适配。
- 风险：valid ratio 偏低，说明该 reward 不能单独解决重建合法性。

### sa

- 终点：valid=0.562，fragment=0.347，dock=-7.856，LE=0.367。
- 分子性质终点：logP=0.290，TPSA=105.740，carbon=0.650，hetero=7.784，heavy=22.580。
- 环结构终点：rings=2.556，benzene=0.062，max fused=1.679，pathology=0.556。
- 训练变化：valid -0.069，logP -0.186，TPSA +1.876，carbon -0.012，pathology +0.061。
- 解读：SA 主要约束合成可及性，理论上会压过复杂结构；如果 valid 或 docking 明显下降，说明它可能把生成方向推得过于保守或简单。
- 风险：终点仍有较高环病态比例，单独作为结构质量控制不足。
- 风险：valid ratio 偏低，说明该 reward 不能单独解决重建合法性。

### tpsa_distribution

- 终点：valid=0.653，fragment=0.250，dock=-7.911，LE=0.363。
- 分子性质终点：logP=0.282，TPSA=109.743，carbon=0.645，hetero=7.921，heavy=23.032。
- 环结构终点：rings=2.564，benzene=0.043，max fused=1.660，pathology=0.532。
- 训练变化：valid -0.028，logP -0.151，TPSA +4.573，carbon -0.014，pathology +0.042。
- 解读：TPSA distribution 是极性分布约束，不是 TPSA 越高越好。它的价值在于防止极性坍缩或低 TPSA 漂移，但单独使用时可能缺少靶点选择压力。
- 风险：终点仍有较高环病态比例，单独作为结构质量控制不足。

### ecfp4_prior

- 终点：valid=0.618，fragment=0.306，dock=-7.708，LE=0.360。
- 分子性质终点：logP=0.241，TPSA=110.407，carbon=0.639，hetero=7.968，heavy=22.517。
- 环结构终点：rings=2.483，benzene=0.034，max fused=1.607，pathology=0.494。
- 训练变化：valid -0.069，logP -0.381，TPSA +4.037，carbon -0.031，pathology -0.021。
- 解读：ECFP4 prior 更偏结构骨架相似性，能把分子拉回 ChEMBL 常见结构空间；但如果只用它，可能优化相似度而不是靶点性能。

### fcfp4_prior

- 终点：valid=0.646，fragment=0.264，dock=-7.839，LE=0.354。
- 分子性质终点：logP=0.329，TPSA=113.208，carbon=0.643，hetero=8.159，heavy=23.387。
- 环结构终点：rings=2.441，benzene=0.032，max fused=1.667，pathology=0.430。
- 训练变化：valid -0.056，logP -0.386，TPSA +4.747，carbon -0.024，pathology -0.085。
- 解读：FCFP4 prior 更偏功能团/药效团环境，相比 ECFP4 更关注原子类型和局部化学特征；它是否保护杂原子和 TPSA 是重点。

### logp_distribution

- 终点：valid=0.660，fragment=0.229，dock=-7.774，LE=0.356。
- 分子性质终点：logP=0.423，TPSA=106.986，carbon=0.648，hetero=7.829，heavy=23.137。
- 环结构终点：rings=2.589，benzene=0.032，max fused=1.821，pathology=0.474。
- 训练变化：valid -0.035，logP -0.245，TPSA -0.248，carbon -0.022，pathology -0.026。
- 解读：LogP distribution 直接约束疏水性，是防全碳/高疏水漂移的重要候选。单独使用时要看是否牺牲 docking 和有效率。

### ligand_efficiency

- 终点：valid=0.688，fragment=0.208，dock=-7.768，LE=0.349。
- 分子性质终点：logP=-0.085，TPSA=117.058，carbon=0.630，hetero=8.515，heavy=23.293。
- 环结构终点：rings=2.657，benzene=0.040，max fused=1.758，pathology=0.465。
- 训练变化：valid +0.042，logP -0.679，TPSA +11.825，carbon -0.037，pathology -0.041。
- 解读：Ligand efficiency 用 docking/重原子数约束大分子 shortcut。它比 docking 更贴近实际候选优化，但单独使用可能偏向小分子或对失败样本过敏。

### chembl_classifier

- 终点：valid=0.632，fragment=0.285，dock=-7.605，LE=0.366。
- 分子性质终点：logP=0.462，TPSA=102.297，carbon=0.650，hetero=7.486，heavy=22.143。
- 环结构终点：rings=2.604，benzene=0.022，max fused=1.846，pathology=0.473。
- 训练变化：valid -0.049，logP -0.079，TPSA +0.989，carbon -0.024，pathology -0.027。
- 解读：classifier 是人工经验约束的代理。它能否作为 reward，取决于是否减少坏分子而不误杀正常结构；单独训练尤其要警惕 classifier reward hacking。


## 7. 对下一步 reward 设计的启发

1. 不建议直接把 docking、classifier、分布约束、guard 全部混在一起训练。九组结果显示，每个单奖励都会带来不同副作用；混合后很难判断是哪个变量造成漂移。
2. DiffSBDD 的核心风险仍然是底座合法性和重建链条：多数组的 RDKit valid ratio 只有约 0.5-0.7，fragmented molecule 比例不低。奖励函数可以改变分布，但不能完全修复点云到键重建的问题。
3. 对最终组合 reward，建议每轮只增加一个模块：先确定 docking 或 ligand efficiency 作为性能驱动，再单独加一个分布约束或结构约束，观察是否牺牲有效率和 docking。
4. 如果目标是防止全碳/疏水漂移，优先考虑 `logp_distribution` 或 `tpsa_distribution` 这类分布约束，而不是只靠 classifier。classifier 更适合作为后期人工经验过滤或弱辅助，不适合作为唯一主 reward。
5. 如果目标是保持药物样和可合成性，QED/SA 的方向相对温和，但它们对靶点 docking 的直接贡献弱，需要和性能项组合验证。
6. KL=0.05 在本轮未表现为主要训练瓶颈：`KL_Penalty_Weighted` 数值整体较小，说明当前更大的问题是 reward 对分子分布的选择压力和底座重建质量，而不是 KL 过强压住训练。

## 8. 建议的下一轮最小组合测试

| 组合 | 目的 | 预期观察 |
|---|---|---|
| docking + logp_distribution | 测试能否保 docking 同时压疏水漂移 | docking 是否下降、logP/carbon 是否回落 |
| docking + tpsa_distribution | 测试能否保 docking 同时维持极性 | TPSA/HBA/HBD 是否稳定，有效率是否下降 |
| ligand_efficiency + logp_distribution | 测试是否避免大分子和高疏水 shortcut | heavy atoms、logP、docking/LE 三者关系 |
| docking + chembl_prior_ecfp4 | 测试结构先验是否改善可用骨架 | ChEMBL 相似度是否提升、fragment 是否改善 |
| docking + classifier | 只作为人工经验方向验证 | 是否出现 classifier reward hacking 或误杀正常分子 |

下一轮仍建议每次只改变一个变量，例如先固定 docking 权重，再只加入一个约束模块，避免训练后无法归因。
