# DiffSBDD E2PO 全实验总审计

> **命名说明**：本文档原名 `DiffSBDD E2PO 全实验总审计`。**E2PO 已于 2026-09-18 更名为 ESRPO**
> （Endpoint Structural-Rule Preference Optimization），`e2po` 保留为兼容别名
> （更名记录见 [260918-02](../../work_records/26-9/260918/260918-02 E2PO更名为ESRPO全面更新.md)）。本文正文保留当时的 E2PO 用词。
> ⚠️ **本文正文截至 2026-09-02**；之后的参照重算、ChEMBL 对接与跨源版本见 **文末 §11「后续进展」**。
> **相关**：[`DiffSBDD-ESRPO-ChEMBL版总审计.md`](DiffSBDD-ESRPO-ChEMBL版总审计.md)（ESRPO 的 ChEMBL 版本，winner 取自 ChEMBL）。

## 一句话结论

截至 2026-09-02，DiffSBDD endpoint E2PO 不能概括为“模型完全学不到”：它已经证明可以拟合 endpoint 去噪偏好，并在部分实验中改善独立生成的 docking 和 ligand efficiency；但当前最强拟合主要来自 `v` 对元素组成的 shortcut，结构匹配 `x_only` 只在低/中 timestep 学到可测信号，而训练 pair accuracy 与真实生成质量多次脱钩。因此，E2PO 目前是一个**尚未通过生成闭环的候选研究分支**，不是已验证主线，也不是完整 reverse-trajectory DPO 或逐 timestep credit assignment 方法。

![E2PO 训练代理、时间步信号和生成风险总览](../../work_records/images/e2po_evidence_overview.png)

图中每一点均为现有单次 run 的确定性汇总，不是多 seed 均值；没有误差线是因为大多数实验只有一个训练 seed。完整训练总表、生成证据表和路线决策矩阵位于：

- [训练实验总表](../../../analysis/diffsbdd_e2po_master_audit_260902/training_experiment_ledger.csv)
- [生成证据总表](../../../analysis/diffsbdd_e2po_master_audit_260902/generation_evidence_ledger.csv)
- [下一步决策矩阵](../../../analysis/diffsbdd_e2po_master_audit_260902/experiment_decision_matrix.csv)

## 1. 这套方法实际上是什么

### 1.1 本地实现的真实计算

对 winner 和 loser endpoint，在一个抽样 diffusion timestep 上加入 ligand-only 噪声，使用 clean pocket conditioning，计算当前模型与冻结 reference 模型的 endpoint 去噪误差：

```text
score(endpoint) = -(w_x * MSE_x + w_v * MSE_v)
current_margin   = score_current(winner) - score_current(loser)
reference_margin = score_reference(winner) - score_reference(loser)
relative_margin = current_margin - reference_margin
```

普通 Global DPO 使用 winner/loser 硬标签；E2PO 使用 frozen endpoint reward 经过 softmax 得到软标签：

```text
p_winner = softmax(temperature * [reward_winner, reward_loser])[0]
```

因此，validation/test 的 `pair_accuracy` 是 `current_margin > 0` 的比例；它回答“当前模型在该 timestep 的去噪误差是否更偏向 winner”，不等同于：

- 分子生成正确率；
- docking 成功率或 docking 分数；
- ChEMBL 相似度；
- 结构风险分类器准确率；
- current 相对于 reference 的相对偏好准确率。

### 1.2 方法边界

| 问题 | 本地 E2PO 能否直接回答 |
|---|---|
| 终点分子 winner 是否得到更高 endpoint 去噪代理分数 | 能 |
| 整条 reverse trajectory 的精确 log-likelihood 是否更高 | 不能；当前不是 full-trajectory likelihood DPO |
| 每个 timestep 的动作或状态应得到什么 credit | 不能；一个 endpoint reward 被广播到抽样 timestep |
| 是否完全复现 AliDiff | 不能；这是 DiffSBDD-native adaptation |
| 是否能保证生成分子 docking/结构/化学空间改善 | 不能；必须固定 attempts 独立生成后验证 |

实现证据：

- `/data/disk6/DiffB110/ddpo/global_dpo/global_score.py`
- `/data/disk6/DiffB110/ddpo/global_dpo/loss.py`
- `/data/disk6/DiffB110/ddpo/global_dpo/paired_noise.py`
- `/data/disk6/DiffB110/ddpo/global_dpo/trainer.py`

## 2. 实验谱系总览

### 2.1 可比训练结果

| 实验 | 数据/reward | 训练设置 | test pair acc | 生成闭环 | 定性结论 |
|---|---|---:|---:|---|---|
| 早期 Global DPO T=1000 | binary good/bad | 5 epoch, lr 1e-6 | 0.6172 | 每 epoch 1000 attempts | pair 拟合上升但生成严重崩坏 |
| 固定 gap E2PO T=500 | reward gap 恒 2 | 5 epoch, lr 1e-6 | 0.6178 | 无 | 只验证 fixed-soft-label 接口 |
| strong docking `x_only` | docking strong good/bad | 4096 pairs, lr 1e-6 | 0.5354 | 1000 attempts | 分布保守，几乎无 docking 增益 |
| strong docking `v_only` | 同上 | 同上 | 0.9732 | 1000 attempts | 极强 shortcut；docking 增益伴随高碳漂移 |
| strong docking native-joint | 同上 | 同上 | 0.8028 | 1000 attempts | docking/LE 最好，但 guard 和组成恶化 |
| ECFP4 native-joint | ChEMBL ECFP4 | 4096 pairs, lr 1e-6, temp 5 | 0.5717 | 1000 attempts | ChEMBL/docking 小幅增益，仍有组成漂移 |
| matched structure `x_only` 低预算 | R1-R4, composition matched | 4096 pairs, lr 1e-6 | 0.5257 | 1000 attempts | 旧 checkpoint 生成 R1-R5 无变化 |
| matched structure `x_only` 全量 | R1-R4, composition matched | 36232 pairs, lr 1e-5 | 0.5921 | **无** | 低 t 明确学到，最新模型尚未生成验证 |
| matched structure `x:v=10:1` | 同上 | 36232 pairs, lr 1e-5 | 0.5486 | 无 | 非归一化尺度混杂且可能大量 clipping |
| docking+ChEMBL 50:50 v2 | percentile 50:50 | 6532 pairs, lr 1e-5 | 0.5491 | **无** | 软标签弱且 32% pair 逆转结构方向 |

这张表不能只按 pair accuracy 排名。早期 Global DPO 和 strong docking `v_only` 已经构成两个反例：代理拟合更强，生成却可能更差或沿 shortcut 漂移。

## 3. 各类实验的证据和含义

### 3.1 早期 Global DPO：最明确的反例

路径：

`/data/disk6/DiffB110/analysis/diffsbdd_global_dpo_t1000_v1/formal_5epoch_native_joint_1000gen`

5 epoch 后 validation/test pair accuracy 为 `0.6021/0.6172`，但生成从 Original 到 epoch 5 出现：

| 指标 | Original | epoch 5 |
|---|---:|---:|
| sanitize valid / 1000 | 673 | 816 |
| unique Murcko scaffold | 600 | 163 |
| carbon fraction | 0.740 | 0.077 |
| TPSA | 62.56 | 294.18 |
| QED | 0.613 | 0.075 |
| fragments mean | 1.10 | 3.84 |
| Brenk hit | 58.40% | 99.88% |

有效率升高并不等于分子更好；该模型主要把分布推向极端富杂原子、多碎片和低 QED 区域。它证明：**pair accuracy 和 relative margin 可以持续增加，同时生成质量持续恶化。**

### 3.2 固定 gap pilot：接口成功，不是连续 reward 成功

路径：

`/data/disk6/DiffB110/analysis/diffsbdd_global_e2po_t500_v1/formal_run_gpu5`

- 17,289 train pairs，5 epoch；
- reward gap 的 min/max/mean 全为 `2.0`，unique count 为 `1`；
- winner weight 固定为 `0.880797`；
- test pair accuracy `0.617808`；
- t50 到 t500 为 `0.7091, 0.6606, 0.6111, 0.5802, 0.5281`。

这证明 reward-sidecar 到 soft-label loss、训练、验证和 checkpoint 的链路可工作，但没有检验不同 reward gap 是否被正确排序，也没有生成端证据。

### 3.3 strong docking x/v：`v` 能学，但主要学到 shortcut

路径：

`/data/disk6/DiffB110/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825`

训练条件固定为 4096 pairs、1 epoch、2048 updates、lr `1e-6`、independent noise。固定 1000 attempts 生成结果：

| 方法 | test pair acc | valid/1000 | docking median | LE median | carbon fraction | TPSA | guard violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | - | 653 | -9.640 | 0.455 | 0.747 | 61.94 | 40.4% |
| x-only | 0.535 | 671 | -9.713 | 0.457 | 0.747 | 61.97 | 44.9% |
| v-only | 0.973 | 711 | -10.321 | 0.483 | 0.833 | 39.24 | 63.7% |
| native-joint | 0.803 | 692 | -10.356 | 0.485 | 0.823 | 40.52 | 60.7% |

结论不是“`v` 是坏的”，而是当前 strong docking pair 的 winner/loser 在组成上可分，`v` 分支能快速利用 atom-feature/composition 通道。结果确实改善 docking 和 LE，但高碳、低 TPSA、较高 LogP、较高 chemistry-guard violation 同时出现。对 native-joint 的进一步定量分析还显示：

- topology burden index 相对 Original `+0.097` 个 Original SD，95% bootstrap CI `[+0.034,+0.160]`；
- large-ring rate `33.84% -> 40.32%`；
- guard violation `40.43% -> 60.69%`；
- UMAP centroid shift `0.4317`，spread area ratio `0.866`；
- native-joint 比 Original 更远离 ChEMBL reference 主体中心。

所以“高碳”在这里代表组成更疏水、低极性，不代表拓扑更简单。若用户接受少量高碳漂移，可以把它当作可容忍副作用，但不能把 `0.823` carbon fraction 和 `+20.3` percentage-point guard 漂移称为“少量”。

### 3.4 ECFP4 reward：存在小幅真实信号，但效应小且未多 seed

路径：

`/data/disk6/DiffB110/analysis/diffsbdd_global_e2po_ecfp4_260825`

训练 test pair accuracy `0.5717`。固定 1000 attempts：

| 指标 | Original | ECFP4-E2PO |
|---|---:|---:|
| valid/1000 | 653 | 708 |
| ECFP4 top-5 mean median | 0.2518 | 0.2579 |
| docking clean median | -9.665 | -9.983 |
| LE median | 0.455 | 0.472 |
| carbon fraction | 0.747 | 0.805 |
| TPSA | 61.94 | 48.87 |
| LogP | 2.72 | 3.58 |
| Brenk alert | 57.9% | 54.2% |

按 Original carbon-fraction 分布重加权后，ECFP4 top-5 mean 的差值没有消失；因此不能把总体 ECFP4 增益简单归因于高碳。不过按 carbon fraction 最近邻精细匹配后的差异接近 0，说明当前观察仍不足以作因果归因。合理表述是：**单 seed 下有小幅 ECFP4 和 docking 联合增益，同时出现中等组成漂移；值得作为辅助证据，不足以成为主线成功结论。**

### 3.5 matched structure `x_only`：不是完全学不到，而是低 t 学、高 t 不学

pair 路径：

`/data/disk6/DiffB110/analysis/diffsbdd_structure_pairs_matched_260831`

构造约束：节点数和元素组成完全匹配，heavy atom/hetero atom 相同，docking 均值差接近 0；winner 的 R1-R4 structural violation 为 0，loser 约 2；R5 高碳护栏匹配且不参与偏好标签。train/validation/test 为 `36,232/3,495/3,480` pairs。

低预算 4096 pairs、lr `1e-6` 的 test accuracy 只有 `0.5257`；全量 36,232 pairs、lr `1e-5` 后提升到 `0.5921`。更重要的是分 timestep：

| 模型 | t50 | t125 | t250 | t375 | t500 |
|---|---:|---:|---:|---:|---:|
| 低预算 x-only | 0.585 | 0.549 | 0.525 | 0.464 | 0.506 |
| 全量 lr1e-5 x-only | 0.712 | 0.636 | 0.584 | 0.529 | 0.499 |

这证明三个事实：

1. 增加 `lr × updates` 确实让 `x_only` 学到了结构偏好，不是模型完全无能力；
2. 结构信息在低/中噪声 timestep 最清晰，t500 仍接近随机；
3. 用五个 timestep 平均得到的 `0.592` 会掩盖 t50 的强信号和 t500 的无信号。

旧低预算 checkpoint 已生成 1000 attempts：Original 和 trained 的 R1-R5 pathology 分别为 `362/653=55.44%` 和 `374/675=55.41%`，基本无变化；若只看 R1-R4，分别为 `334/653=51.15%` 和 `335/675=49.63%`，是 `-1.52` percentage points 的小幅描述性差异，尚无多 seed 或显著性支持。**这一旧结果不能否定最新全量 lr1e-5 checkpoint，因为后者尚未生成。**

最新 checkpoint：

`/data/disk6/DiffB110/analysis/diffsbdd_global_e2po_structure_matched_x_only_lr1e5_full_260901/checkpoints/diffsbdd-global-dpo-epoch=0001.pt`

### 3.6 `x:v=10:1`：当前不能解释为公平的“少量 v”消融

路径：

`/data/disk6/DiffB110/analysis/diffsbdd_global_e2po_structure_matched_xv10_1_lr1e5_full_260901`

实际配置是 `x_weight=10.0, v_weight=1.0`，而不是归一化的 `10/11,1/11`。因此它同时改变：

- x/v 比例；
- total score 尺度，约放大 11 倍；
- relative margin 和梯度尺度；
- 在 `max_grad_norm=1.0` 下触发 clipping 的概率。

其 mean gradient norm 为 `5.43`，test pair accuracy `0.5486`，低于 x-only `0.5921`。这个结果只能说“未归一化的 10,1 配置不好”，不能证明“加入 9.1% 的 v 一定有害”。后续 x/v 比较必须保持 `x_weight+v_weight=1`，例如 `1.0/0.0`、`0.9/0.1`、`0.7/0.3`、`0.5/0.5`。

### 3.7 docking+ChEMBL 50:50 v2：当前目标函数和 pair 本身都偏弱

pair 路径：

`/data/disk6/DiffB110/analysis/diffsbdd_global_e2po_docking_chembl_50_50_pairs_260901_v2`

训练路径：

`/data/disk6/DiffB110/analysis/diffsbdd_global_e2po_docking_chembl_50_50_lr1e5_full_260901`

设计是 `0.5*docking percentile + 0.5*ECFP4 percentile`，接受 pair 要求 docking gap 至少 0.3、ChEMBL drop 不超过 0.03、combined gap 至少 0.10。结果为 test accuracy `0.5491`，t50 到 t500 为 `0.597,0.599,0.549,0.510,0.490`。

不能只归因于训练 pair 少，原因至少有两个：

1. combined reward gap 均值只有 `0.2432`，temperature 为 1，winner soft weight 均值仅 `0.5603`；标签非常接近 50:50，DPO 梯度天然弱；
2. pair 是从 structure-matched pair 重新定向的。train/validation/test 中分别有 `2092/6532`、`286/892`、`288/896`，约 `32%` pair 被翻转，即原 R1-R4 pathological endpoint 因 docking+ChEMBL 更好而成为 winner。

所以这不是“docking+ChEMBL 且结构安全”的训练，而是“允许约三分之一 pair 用 docking+ChEMBL 覆盖结构安全方向”。该 checkpoint 尚未生成和重 dock，当前不能判断最终分子效果；但不应直接扩 epoch，因为 pair 目标冲突和温度尺度不会通过增加 epoch 自动消失。

## 4. 为什么学不到，或者学到的东西不好

### 4.1 endpoint MSE 代理和生成目标之间有距离

当前 score 是单 timestep endpoint epsilon-MSE，不是完整 reverse trajectory 的精确 likelihood，也不直接优化最终 OpenBabel reconstruction 后的分子、ring topology、docking pose 或 ChEMBL fingerprint。因此出现以下脱钩并不意外：

```text
endpoint pair margin improved
        does not imply
reverse sampling probability moved in the desired way
        does not imply
reconstructed molecule passes chemistry/structure gates
        does not imply
docking and ChEMBL both improve
```

### 4.2 timestep 信噪比不均匀

matched structure `x_only` 的 t50 为 `0.712`，t500 为 `0.499`；50:50 的 t50/t125 约 `0.60`，t500 为 `0.49`。均匀从 `1..500` 采样会把高信息低 t 和低信息高 t 混合。高 t 不一定始终是反信号，但至少当前数据证明其边际贡献很弱。继续只加 epoch 可能把计算花在低信噪比区间，而不会等比例提高生成结构质量。

### 4.3 `x` 和 `v` 承载不同捷径

- `x` 对结构几何更保守，matched pair 后在低 t 能学习，但总体信号弱；
- `v` 直接包含 atom-feature denoising 通道，当 winner/loser 组成有差异时极易把 composition 当标签；
- native-joint 中 `v_weight=0.769`，因此 strong docking 实验主要呈现 v-driven 行为。

### 4.4 pair 构造决定模型“学到什么”

- strong docking good/bad：可用组成 shortcut，模型迅速学高碳、低极性；
- matched structure：shortcut 被堵住，任务更难，但更接近真正结构风险；
- docking+ChEMBL v2：约 32% pair 与 R1-R4 安全方向冲突；
- fixed gap pilot：所有 pair 软标签一样，只能学固定二分类方向。

所以 pair 数量不是唯一的“数据量”。pair 的可辨识信息、冲突率、奖励间隙和混杂控制更重要。

### 4.5 E2PO reward 的数值尺度会改变有效监督强度

| 实验 | reward temperature | mean gap | mean winner weight |
|---|---:|---:|---:|
| strong docking | 1 | 5.640 | 0.996 |
| structure full | 1 | 1.978 | 0.842 |
| ECFP4 | 5 | 0.209 | 0.736 |
| docking+ChEMBL 50:50 | 1 | 0.243 | 0.560 |

同样叫 E2PO，监督强度差别极大。50:50 的 loss `0.69283` 接近 `ln2`，不是单纯“模型笨”，而是标签本身非常软。temperature 应依据 gap 分布校准到预定 winner-weight 区间，并在报告中固定下来。

### 4.6 训练预算有作用，但不是万能解释

matched `x_only` 从 4096 pairs、lr1e-6、2048 updates 到 36232 pairs、lr1e-5、18116 updates，test 从 `0.526` 到 `0.592`，证明预算有作用。但如果 pair 目标冲突、temperature 太低、timestep 大量无信号，单纯更多 epoch 只会更强地拟合错误或模糊目标。

### 4.7 I/O 是效率瓶颈，不是科学失败原因

低预算 x-only 2048 updates 耗时约 54 分钟，主要因 endpoint shard 随机读取和评估重复前向；全量 lr1e-5 的训练 elapsed 约 2509 秒。cache/预解包可以显著缩短迭代时间，但不会自动修复 proxy、pair 冲突或生成脱钩。

## 5. 目前哪些结论可以成立

### 5.1 可以成立

1. 本地 E2PO 训练链路可以稳定完成，并能学到 held-out endpoint preference。
2. `v` 对 strong docking pair 的拟合能力极强，但存在明确 composition shortcut。
3. native-joint 和 v-only 在当前单 seed 生成中改善 docking/LE，同时显著改变 carbon/TPSA/guard 分布。
4. ECFP4-E2PO 在单 seed 下有小幅 ECFP4、validity、docking 和 LE 增益。
5. matched structure `x_only` 全量 lr1e-5 在低/中 timestep 学到真实 held-out 信号。
6. 当前 50:50 v2 的 soft label 偏弱，而且约 32% pair 与结构安全方向相反。

### 5.2 不能成立

1. 不能说 E2PO 已经总体提升分子质量。
2. 不能用 pair accuracy 替代生成、结构、docking 或 ChEMBL 评价。
3. 不能说 `v_only=0.973` 证明 v-head 学到了普适 docking 物理规律。
4. 不能说 matched `x_only` 全量模型无效，因为最新 checkpoint 尚未生成。
5. 不能说 10:1 证明少量 v 有害，因为 score 未归一化且 clipping 混杂。
6. 不能说当前 50:50 是结构安全的 docking+ChEMBL 中和奖励。
7. 不能说 endpoint E2PO 解决了 trajectory credit assignment。

## 6. 现在应该怎么做

### 6.1 第一优先级：先闭环最新 structure `x_only`，不要再训练

直接使用已完成的全量 lr1e-5 checkpoint，固定 `1000 attempts` 生成，然后和同 seed Original 以固定 attempts 分母比较：

- SDF records / 1000；
- RDKit sanitize-valid / 1000；
- R1-R4 分项和合计，**不要把 R5 混入结构主指标**；
- R5 carbon guard；
- Brenk、fragment、bad bond、large ring、four-member ring、bad three-member ring；
- carbon fraction、hetero atom、TPSA、LogP、MW；
- Murcko scaffold、unique SMILES；
- independent docking、LE、ECFP4 top-5 mean。

放行标准不是 test `0.592`，而是：R1-R4 相对 Original 下降，同时 validity、docking、ChEMBL 和组成没有不可接受回退。若 R1-R4 不下降，说明 endpoint x-MSE proxy 没有形成生成闭环，应停止结构 E2PO 主攻，而不是继续堆 epoch。

### 6.2 第二优先级：做 low-t 与 uniform 的受控消融

若最新 checkpoint 的生成有初步 R1-R4 改善，再做：

| 项目 | control | low-t arm |
|---|---|---|
| pair | 同一 matched R1-R4 | 同一 |
| x/v | 1/0 | 1/0 |
| lr、updates、seed | 完全相同 | 完全相同 |
| timestep | uniform 1..500 | 1..200 或离散 50/125/250 |
| generation | 同 seed 1000 attempts | 同 seed 1000 attempts |

主问题是低 t 信号能否转成更低的生成 R1-R4，不是 pair accuracy 是否继续升高。

### 6.3 第三优先级：重新构造 docking+ChEMBL，不继续当前 v2

新 pair 不应从 clean/pathological 对中自由翻转，而应满足硬约束：

```text
docking(winner) < docking(loser)
ECFP4(winner) >= ECFP4(loser) - small_tolerance
R1-R4(winner) <= R1-R4(loser)
prefer both endpoints R1-R4 clean
composition / heavy atoms / hetero atoms / TPSA / ring topology matched
```

50:50 percentile 可保留为 pair 内排序分数，但结构安全应是 hard constraint，不是又一个可被平均掉的 0.5 reward。依据 gap 分布校准 temperature，使 mean winner weight 处在预注册区间，例如 `0.70-0.80`；这个区间是实验设计目标，需要先固定后训练，不应事后挑选。

首轮使用归一化 `x:v=0.7:0.3` 或 `0.5:0.5`，所有权重和为 1。先小预算训练，再固定 1000 attempts。只有 docking/LE 提升且 R1-R4、validity、R5、TPSA、ChEMBL 都过门槛，才扩量。

### 6.4 强 docking v-head 只保留为机理对照

当前 v-only/native-joint 证明“生成分布能被推向更好 docking”，这是有价值的；但它们不应直接成为正式模型。只有在 composition/topology 严格匹配的 pair 上，docking 增益仍然存在且 guard 不恶化，才能证明不是高碳 shortcut。

## 7. 预注册止损线

| 检查点 | 放行 | 止损 |
|---|---|---|
| 最新 full x-only 生成 | R1-R4 明确下降，其他门槛可接受 | R1-R4 不下降：停止 structure endpoint E2PO |
| low-t 消融 | proxy 和生成 R1-R4 同向改善 | proxy 升、生成不变：判定 proxy-generation 脱钩 |
| 重建联合 pair | 无结构逆向 pair，winner weight 校准，held-out > baseline | pair 仍冲突或 soft label 接近 0.5：不训练 |
| 联合生成 | docking/LE 改善且结构、validity、ChEMBL 不回退 | 任一核心 guard 明显恶化：不升级 |
| strong v 分支 | matching 后 docking gain 仍在 | matching 后 gain 消失：确认 composition shortcut |

## 8. 当前最终路线判断

建议保留两条窄路线：

1. **结构安全主验证线**：matched R1-R4 + x-only + 最新 full checkpoint 生成闭环；只有闭环成功才测试 low-t。
2. **受约束药效线**：重新构造 clean/non-worsening structure 的 docking+ChEMBL Pareto pairs，归一化 x/v 和校准 soft label。

不建议继续：

- 给当前 50:50 v2 直接加 epoch；
- 把 `v_only` 或 strong-docking native-joint 直接作为正式模型；
- 用未归一化 `10,1` 继续做 x/v 比较；
- 仅因 pair accuracy 高就进入全量生成；
- 仅因 ChEMBL 点云接近或 docking 更好就忽略结构风险。

如果用户的真实战略偏好是“允许少量高碳漂移，以换 docking 提升”，这是可行的多目标取舍，但必须预先把“少量”量化，例如 carbon fraction、R5、TPSA 和 guard 的最大允许变化；不能在看到结果后再调整容忍线。当前 native-joint 的 carbon fraction `+0.076`、TPSA `-21.4`、guard `+20.3 pp` 已经是明显分布迁移，而不是轻微扰动。

## 9. 证据路径

### 总审计产物

- `/data/disk6/DiffB110/analysis/diffsbdd_e2po_master_audit_260902/training_experiment_ledger.csv`
- `/data/disk6/DiffB110/analysis/diffsbdd_e2po_master_audit_260902/generation_evidence_ledger.csv`
- `/data/disk6/DiffB110/analysis/diffsbdd_e2po_master_audit_260902/experiment_decision_matrix.csv`
- `/data/disk6/DiffB110/analysis/diffsbdd_e2po_master_audit_260902/e2po_evidence_overview.png`
- `/data/disk6/DiffB110/analysis/diffsbdd_e2po_master_audit_260902/plot_master_audit.py`

### 关键历史报告

- `/data/disk6/DiffB110/work_records/26-8/260825/260825-5 DiffSBDD Global E2PO四组实验完整报告.md`
- `/data/disk6/DiffB110/work_records/26-8/260829/260829-2 定量研究E2PO结构复杂度与xv权重.md`
- `/data/disk6/DiffB110/work_records/26-8/260831/260831-04 条件匹配与单卡E2PO测试状态.md`
- `/data/disk6/DiffB110/work_records/26-9/260901/260901-01 匹配结构x_only单卡学习测试.md`
- `/data/disk6/DiffB110/work_records/26-9/260901/260901-02 matched-x_only生成1000个.md`

## 10. 审计边界

- 本报告没有启动训练、生成或 docking；只读取现有代码和结果，并补做确定性表格汇总、pair 方向审计和旧 SDF 的 R1-R5 统计。
- 大多数结果是单 pocket、单训练 seed、单 generation seed、1000 attempts；bootstrap 分子级区间不等于训练重复不确定性。
- 早期实验的 Original 基线来源/valid 数有 `653` 和 `673` 两套固定生成批次，报告在各自实验内比较，没有混用分母。
- 最新 full x-only 和 docking+ChEMBL 50:50 均没有生成端闭环，任何生成质量结论均保留。
- 总览图已通过 plot-area 对齐审计和 PDF 最小 5 pt 字号审计；当前 `dif`/系统 Python 均缺少 PyMuPDF，因此 rendered collision audit 未执行，不能宣称该项通过。人工检查 PNG 未见明显标签重叠。

## 11. 后续进展（2026-09-03 之后，本文未收录）

> 本文正文截至 **2026-09-02**。之后的工作改变了部分口径与参照，读本文结论前请先看本节：

| 后续工作 | 关键更新 | 记录 |
|---|---|---|
| **R1–R5 参照重算** | 参照集重建为 **1,790,085**（Small molecule + 最大片段 + 中性 + 10 元素白名单，无尺寸窗口）；R1–R5 全清 **84.72% → 80.94%**；**HA10–30 口径 = 90.60%**（生成质量的真实目标线） | [260928-03](../../work_records/26-9/260928/260928-03 ChEMBL新参照集R1-R5合理率重算.md) |
| **ChEMBL 全量对接 RE-CmeB** | 90 片 × 20k 分子，98.42% 成功率，分数中位 **−9.16**（vina/fast，与五模型横评同口径） | [260928-02](../../work_records/26-9/260928/260928-02 ChEMBL小分子参照集：处理方法与构建.md)、`260928-04` |
| **ESRPO-ChEMBL 版（跨源偏好对）** | winner 改为 ChEMBL 真实分子（unidock pose）；证明 **HA 匹配必须**、揭示了“负向规则奖励 → 简单化”机制与 loser 定义漏洞 | `docs/overview/DiffSBDD-ESRPO-ChEMBL版总审计.md` |
| **更名** | E2PO → **ESRPO**（Endpoint Structural-Rule Preference Optimization），`e2po` 保留为兼容别名 | [260918-02](../../work_records/26-9/260918/260918-02 E2PO更名为ESRPO全面更新.md) |
| 评估口径提醒 | 五模型横评/step4000 的生成是 `resamplings=0`，主线生成是 `resamplings=10`，两者“合理分子对接中位”**不可直接横比**（幸存者偏差不同） | `docs/overview/DiffSBDD-ESRPO-ChEMBL版总审计.md` §4.4 |
