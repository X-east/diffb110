# DiffSBDD DPO 支线总审计（Global DPO 硬标签家族）

> **命名说明**：本文写于 2026-09-07，正文使用旧名 **E2PO**；该方法已于 2026-09-18 更名为
> **ESRPO**（`e2po` 保留为兼容别名，[260918-02](../work_records/26-9/260918/260918-02 E2PO更名为ESRPO全面更新.md)）。
> 相关：[`DiffSBDD-ESRPO全实验总审计.md`](DiffSBDD-ESRPO全实验总审计.md)。

## 一句话结论

截至 2026-09-07，DiffSBDD DPO 支线（总账 `family=Global_DPO`、硬标签 binary good/bad）共 6 个 run，全部是 **endpoint denoising surrogate DPO**，不是 exact Gaussian trajectory-DPO。它证明：

1. Global DPO 训练链路可稳定完成，并能拟合硬标签 endpoint 偏好（test pair accuracy 0.58–0.62）；
2. 但 pair 拟合与真实生成质量**严重脱钩**——`global_dpo_t1000_epoch5` 是本项目最明确的反例：pair accuracy 升至 0.617，而生成分子碳比例从 0.752 崩到 0.067、QED 从 0.613 崩到 0.075；
3. `frozen_front_*` 5 个变体仅做 x/v 诊断、未做生成闭环，且其中两个 "dominant" 变体实际用了非归一化极值权重（与 E2PO `xv10_1` 同样的 score-scale 混杂）。

因此，DiffSBDD DPO 支线是 **E2PO 软标签之前的硬标签可行性/诊断线**，本身不是已验证主线，也不能作为正式生成模型。E2PO 是在此基础上引入 continuous soft label 的后续衍生线。

## 1. 这套方法实际上是什么

### 1.1 本地实现的真实计算（普通 Global DPO）

对 winner 和 loser endpoint，在同一个抽样 diffusion timestep 上加入 ligand-only 噪声，使用 clean pocket conditioning，计算当前模型与冻结 reference 模型的 endpoint 去噪误差：

```text
score(endpoint) = -(w_x * MSE_x + w_v * MSE_v)
current_margin   = score_current(winner) - score_current(loser)
reference_margin = score_reference(winner) - score_reference(loser)
relative_margin = current_margin - reference_margin
loss            = -log sigmoid(beta * relative_margin)
```

与 E2PO 的关键区别：**硬标签**。Global DPO 的 winner/loser 是二值 `good/bad`（0/1），没有 soft softmax 温度；E2PO 才用 frozen endpoint reward 经 softmax 得到连续软标签。

`pair_accuracy` 仍只回答"当前模型在该 timestep 的去噪误差是否更偏向 winner"，不等同于分子生成正确率、docking 成功率、ChEMBL 相似度或结构风险改善。

### 1.2 方法边界

| 问题 | 本地 DiffSBDD DPO 能否直接回答 |
|---|---|
| 终点分子 winner 是否得到更高 endpoint 去噪代理分数 | 能 |
| 整条 reverse trajectory 的精确 log-likelihood 是否更高 | 不能；当前不是 full-trajectory likelihood DPO |
| 每个 timestep 的动作或状态应得到什么 credit | 不能；一个 endpoint reward 被广播到抽样 timestep |
| 是否能保证生成分子 docking/结构/化学空间改善 | 不能；必须固定 attempts 独立生成后验证 |
| 是否等价于 exact Gaussian trajectory-DPO | 不能；这是 endpoint surrogate DPO |

### 1.3 两条实现路线（重要边界）

`260823-01 T1000 DiffSBDD Gaussian-DPO规范化支线` 记录了一个路线决策：**用户明确选择普通 DiffSBDD Global DPO（endpoint surrogate）作为主线，exact Gaussian trajectory-DPO 仅保留为数学/对照审计，不作为主训练入口。**

| 项目 | 普通 Global DPO（实际训练） | exact Gaussian trajectory-DPO（保留对照） |
|---|---|---|
| 训练输入 | 固定 good/hard-bad endpoint pair | 保存的 `z_t -> z_s` transition |
| noise | 每次训练现场随机采样 `t` 和 `epsilon` | 使用轨迹中保存的状态 |
| score | 全局 endpoint `x+v` denoising epsilon-MSE | Gaussian reverse-transition log-prob |
| 完整轨迹 | 不需要 | 需要 |
| 代码位置 | `ddpo/global_dpo/` | `ddpo/diffusion_dpo/`（仅审计/对照） |
| 当前地位 | 主实现路线 | 保留为数学/对照审计 |

因此本支线 6 个 run 全部来自 `ddpo/global_dpo/` 普通实现；`ddpo/diffusion_dpo/` 的 exact 代码虽然在 260823-01 完成了数学/代码验收（8 passed 数学测试、2-pair smoke、resume smoke 通过），但**未启动任何 2000+ pair 正式训练**，不计入本审计的 run 谱系。

实现证据：

- `/data/disk6/DiffB110/ddpo/global_dpo/endpoint_data.py`
- `/data/disk6/DiffB110/ddpo/global_dpo/paired_noise.py`
- `/data/disk6/DiffB110/ddpo/global_dpo/global_score.py`
- `/data/disk6/DiffB110/ddpo/global_dpo/loss.py`
- `/data/disk6/DiffB110/ddpo/global_dpo/trainer.py`
- `/data/disk6/DiffB110/ddpo/main_diffsbdd_global_dpo.py`

## 2. 实验谱系总览

### 2.1 六个 run 的共同底座

| 配置项 | 值 |
|---|---|
| family | `Global_DPO` |
| reward_or_pair | `binary_good_bad`（硬标签 0/1） |
| base checkpoint | `DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt` |
| timestep_grid | 1000 |
| epochs | 5 |
| learning_rate | 1e-6 |
| train_pairs | 17,289 |
| updates | 43,225 |
| validation / test pairs | 3,548 / 3,111 |
| reward_temperature | 无（硬标签） |

### 2.2 可比训练结果

| 实验 | variant | x_weight | v_weight | test pair acc | mean margin | 生成闭环 | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| `global_dpo_t1000_epoch5` | native_joint | 0.230769 | 0.769231 | 0.617165 | 0.584258 | 每 epoch 1000 attempts | proxy_fit_but_generation_collapsed |
| `frozen_front_balanced` | balanced | 0.5 | 0.5 | 0.597557 | 0.089928 | 无 | xv_diagnostic_only |
| `frozen_front_v_dominant` | v_dominant | 1.0 | 10.0899 | 0.607779 | 0.196082 | 无 | xv_diagnostic_only |
| `frozen_front_v_only` | v_only | 0.0 | 1.0 | 0.608357 | 0.346064 | 无 | xv_diagnostic_only |
| `frozen_front_x_dominant` | x_dominant | 10.0899 | 1.0 | 0.583864 | 0.104025 | 无 | xv_diagnostic_only |
| `frozen_front_x_only` | x_only | 1.0 | 0.0 | 0.581871 | 0.127726 | 无 | xv_diagnostic_only |

t0125 / t0375 分桶（仅这两个桶有记录）：

| 实验 | test_t0125 | test_t0375 |
|---|---:|---:|
| `global_dpo_t1000_epoch5` | 0.608486 | 0.580842 |
| `frozen_front_balanced` | 0.612986 | 0.576985 |
| `frozen_front_v_dominant` | 0.606879 | 0.582771 |
| `frozen_front_v_only` | 0.599807 | 0.582771 |
| `frozen_front_x_dominant` | 0.609772 | 0.574092 |
| `frozen_front_x_only` | 0.609450 | 0.573770 |

**核心观察**：6 个 run 的 test pair acc 全部落在 0.58–0.62，彼此差异极小（< 0.03），说明在硬标签 binary good/bad 设定下，x/v 权重分配、前端冻结与否对 pair 拟合影响很弱。这与 E2PO 软标签线结论一致——代理拟合强弱不能作为生成质量信号。

## 3. 各类实验的证据和含义

### 3.1 `global_dpo_t1000_epoch5`：最明确的生成崩坏反例

路径：

`/data/disk6/DiffB110/analysis/diffsbdd_global_dpo_t1000_v1/formal_5epoch_native_joint_1000gen`

5 epoch 后 validation/test pair accuracy 为 `0.6021 / 0.6172`，但固定 1000 attempts 生成从 Original 到 epoch 5 出现剧烈分布退化（来源：`260824-01 Global DPO生成质量验证与退化诊断`）：

| 指标 | Original | DPO epoch 5 |
|---|---:|---:|
| sanitize valid / 1000 | 673 | 816 |
| unique Murcko scaffold | 600 | 163 |
| carbon fraction | 0.752 | 0.067 |
| LogP | 2.79 | -4.32 |
| TPSA | 62.56 | 294.18 |
| QED | 0.613 | 0.075 |
| fragments mean | 1.10 | 4.12 |
| Brenk hit | 58.40% | 99.88% |
| NIH hit | 28.23% | 99.02% |
| PAINS hit | 3.27% | 21.94% |

validity 上升（673→816）但 scaffold 数从 600 降到 163，说明生成分布越来越集中、而不是更好。描述符显示碳比例极低、杂原子比例极高、TPSA/HBD 过高、环结构消失、碎片数上升——方向与"生成更合适分子"相反。

训练/导出审计确认：每个 checkpoint 有 122 个 state-dict tensor，导出为 DiffSBDD Lightning checkpoint 时 key 和 shape 均匹配；epoch 5 相对原始 checkpoint 的整体参数 L2 相对变化约 0.00330，变化最大的相对参数包括 `ddpm.dynamics.egnn.embedding.bias`、EGNN node MLP bias、`atom_encoder/atom_decoder` bias。因此**没有证据表明是 checkpoint key 错配或权重未加载**，更符合"DPO denoising proxy 被优化，但模型输出分布发生原子类型退化"的现象。

结论：这是本支线最硬的反例——**pair accuracy 和 relative margin 持续增加，同时生成质量持续恶化**。

### 3.2 `frozen_front_*`：x/v 诊断，无生成闭环

路径根目录：

`/data/disk6/DiffB110/analysis/diffsbdd_global_dpo_t1000_frozen_front_xv/{balanced,v_dominant,v_only,x_dominant,x_only}`

设计（来源：`260824-02 固定前端Global DPO多组xv实验`）：固定 `atom_encoder`、`residue_encoder`、`atom_decoder`、`residue_decoder`、`e_block_0..2`，仅让后两层适配偏好；真实 checkpoint 预检确认冻结后仍有 406,424 / 1,005,919 = 40.4% 可训练参数。五组在 5 张 GPU 上各跑一个 variant，设计文档写明生成协议为"每完成 epoch 1000 attempts"。

**审计发现（需标注）**：总账 CSV 记录这 5 个 run 的 `generation_evidence=none`、`status=xv_diagnostic_only`，与设计文档的生成协议不一致——实际未产出生成闭环证据，仅完成训练与 pair 诊断。此外，设计文档列出的变体是 `(1,0),(0.75,0.25),(0.5,0.5),(0.25,0.75),(0,1)`，但总账实际落地的是：

- `x_only` = (1.0, 0.0) ✓ 与设计一致
- `v_only` = (0.0, 1.0) ✓ 与设计一致
- `balanced` = (0.5, 0.5) ✓ 与设计一致
- `x_dominant` = (10.0899, 1.0) ✗ 设计为 (0.75, 0.25)
- `v_dominant` = (1.0, 10.0899) ✗ 设计为 (0.25, 0.75)

即两个 "dominant" 变体实际用了**非归一化极值权重 10.0899**，而非设计中的 0.25/0.75。这与后来 E2PO 的 `xv10_1`（x=10, v=1）是同一类 **score-scale + gradient-clipping 混杂**，因此这两个 run 不能公平解释为"少量 v 主导"，只能作为极值诊断。它们未做生成也与此混杂和诊断定位一致。

其余 3 个（`x_only` / `v_only` / `balanced`）是干净的 x/v 权重对照：

- `x_only` test acc 0.5819，mean margin 0.1277；
- `v_only` test acc 0.6084，mean margin 0.3461（v 分支 margin 最大，与 E2PO strong-docking v_only 结论一致：v 在硬标签下也能快速利用 atom-feature 通道）；
- `balanced` test acc 0.5976，mean margin 0.0899（最低，说明 x/v 各半时总信号被稀释）。

## 4. 为什么学不到，或者学到的东西不好

### 4.1 endpoint MSE 代理和生成目标之间有距离

当前 score 是单 timestep endpoint epsilon-MSE，不是完整 reverse trajectory 的精确 likelihood，也不直接优化最终 OpenBabel reconstruction 后的分子、ring topology、docking pose 或 ChEMBL fingerprint。脱钩链：

```text
endpoint pair margin improved
        does not imply
reverse sampling probability moved in the desired way
        does not imply
reconstructed molecule passes chemistry/structure gates
        does not imply
docking and ChEMBL both improve
```

`global_dpo_t1000_epoch5` 的崩坏是这条链断裂的直接证据。

### 4.2 timestep 信噪比不均匀

硬标签下只记录 t0125 / t0375 两个桶，无法像 E2PO matched structure 那样画出 t50→t500 的清晰衰减。但 6 个 run 全部 test acc ≈ 0.58–0.62、且 t0125 普遍高于 t0375 约 0.02–0.04，与 E2PO 的"低 t 信号强、高 t 接近随机"方向一致，提示高 timestep 的偏好信号同样很弱。

### 4.3 `x` 和 `v` 承载不同捷径

- `x` 对结构几何更保守，硬标签下信号弱（x_only mean margin 仅 0.128）；
- `v` 直接包含 atom-feature denoising 通道，硬标签 good/bad 若 winner/loser 组成有差异，极易把 composition 当标签（v_only mean margin 0.346，最高）；
- `native_joint`（global_dpo_t1000_epoch5）中 `v_weight=0.769`，因此崩坏主要表现为原子类型退化。

### 4.4 pair 构造决定模型"学到什么"

硬标签 binary good/bad 的 winner/loser 来自 endpoint 规则（先验 good/hard_bad 极端对），其组成可分性强，`v` 分支能快速利用——这正是 global_dpo_t1000_epoch5 崩坏的根源。后来的 E2PO matched-R1-R4 用 composition-matched pair 堵住该捷径，是支线演进的直接动机。

## 5. 目前哪些结论可以成立

### 5.1 可以成立

1. 本地普通 Global DPO 训练链路可稳定完成，并能拟合硬标签 endpoint 偏好（test acc 0.58–0.62）。
2. `global_dpo_t1000_epoch5` 是最明确的反例：pair accuracy 上升的同时生成分子严重退化（碳比例 0.752→0.067，QED 0.613→0.075）。
3. `v` 分支在硬标签下 margin 最大（0.346），证实 v 通道易被 composition 捷径利用。
4. 6 个 run 的 test acc 差异极小（< 0.03），说明硬标签 binary good/bad 下 x/v 权重、前端冻结对 pair 拟合影响很弱。

### 5.2 不能成立

1. 不能说 DiffSBDD DPO 已经总体提升分子质量（反例明确）。
2. 不能用 pair accuracy 替代生成、结构、docking 或 ChEMBL 评价。
3. 不能说 `v_only=0.608` 证明 v-head 学到普适物理规律。
4. 不能说 `frozen_front_x_dominant/v_dominant` 证明"少量 v/x 主导有害"——它们实际用了非归一化极值 10.0899，score-scale 与 clipping 混杂。
5. 不能把 `frozen_front_*` 的 pair 结果当作生成证据——它们 generation_evidence=none。
6. 不能把 `ddpo/diffusion_dpo/` exact Gaussian-DPO 代码验收当作训练成功——它未启动正式训练。

## 6. 现在应该怎么做

1. **冻结支线训练**：DiffSBDD DPO 硬标签线不应再扩 epoch 或扩量；它的科学作用已明确（可行性证明 + 反例 + x/v 诊断）。
2. **保留为对照基线**：`global_dpo_t1000_epoch5` 与 `frozen_front_*` 应作为 E2PO 软标签线的硬标签对照，而非独立生成模型。
3. **不要直接并入生成流程**：任何后续生成模型必须来自软标签 E2PO（composition-matched pair）或独立验证路径，不能直接用本支线 checkpoint 做正式生成。
4. **exact Gaussian-DPO 维持对照身份**：`ddpo/diffusion_dpo/` 仅作数学/协议审计，不启动主训练，除非用户重新决策。

## 7. 与 E2PO 支线的关系

| 维度 | DiffSBDD DPO（本支线） | E2PO |
|---|---|---|
| label 类型 | 硬标签 binary good/bad（0/1） | 软标签（continuous preference softmax） |
| family | `Global_DPO` | `E2PO` |
| run 数 | 6 | 10 |
| 时间位置 | 先（可行性/诊断） | 后（软标签衍生） |
| 最强反例 | `global_dpo_t1000_epoch5`（生成崩坏） | `v_only`/`native-joint`（高碳 shortcut） |
| 生成闭环 | 仅 1 个 run 有（且崩坏） | 4 组 strong-docking 有（含增益与漂移） |
| 当前地位 | 已闭合的硬标签对照线 | 尚未通过生成闭环的候选研究分支 |

两线共享同一套 `ddpo/global_dpo/` 训练基础设施与 base checkpoint；E2PO 在该基础设施上增加了 soft-label loss 分支。本支线文档与 `DiffSBDD E2PO全实验总审计.md` 并列，互不隶属。

## 8. 证据路径

### 总账来源

- `/data/disk6/DiffB110/analysis/diffsbdd_e2po_master_audit_260902/training_experiment_ledger.csv`（其中 `family=Global_DPO` 的 6 行）

### 关键历史报告

- `/data/disk6/DiffB110/work_records/26-8/260823/260823-01 T1000 DiffSBDD Gaussian-DPO规范化支线.md`（普通 Global DPO vs exact Gaussian-DPO 路线决策、代码验收）
- `/data/disk6/DiffB110/work_records/26-8/260824/260824-01 Global DPO生成质量验证与退化诊断.md`（`global_dpo_t1000_epoch5` 生成崩坏定量表）
- `/data/disk6/DiffB110/work_records/26-8/260824/260824-02 固定前端Global DPO多组xv实验.md`（`frozen_front_*` 设计与 GPU 映射）
- `/data/disk6/DiffB110/work_records/26-8/260822/260822-09 DiffSBDD与Global DPO可行性深度报告.md`（可行性背景）
- `/data/disk6/DiffB110/work_records/26-8/260822/260822-10 T1000 Global DPO梯度审计与短Pilot.md`（早期 pilot）

### 训练/生成产物

- `/data/disk6/DiffB110/analysis/diffsbdd_global_dpo_t1000_v1/formal_5epoch_native_joint_1000gen/`（含 metrics.csv、evaluation/）
- `/data/disk6/DiffB110/analysis/diffsbdd_global_dpo_t1000_frozen_front_xv/{balanced,v_dominant,v_only,x_dominant,x_only}/`

## 9. 审计边界

- 本报告没有启动训练、生成或 docking；只读取现有代码、总账 CSV 与结果报告，并补做确定性表格汇总。
- 大多数结果是单 pocket、单训练 seed、单 generation seed、1000 attempts；bootstrap 分子级区间不等于训练重复不确定性。
- `frozen_front_*` 的 `generation_evidence=none` 与设计文档"每 epoch 1000 attempts"不一致，已在 §3.2 标注；不把未落地的生成协议当作已执行。
- `x_dominant/v_dominant` 实际权重 (10.0899,1)/(1,10.0899) 与设计 (0.75,0.25)/(0.25,0.75) 不一致，已在 §3.2 标注为 score-scale 混杂诊断。
- `ddpo/diffusion_dpo/` exact Gaussian-DPO 仅完成代码/数学验收，未启动正式训练，不计入 run 谱系。
