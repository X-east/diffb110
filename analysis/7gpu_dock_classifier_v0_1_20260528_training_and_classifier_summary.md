# 总结：7gpu-dock-classifier-v0-1 训练与 classifier 分布分析

## 一、动机/背景

本次任务的原始问题是：`7gpu-dock-classifier-v0-1_20260528_000841` 训练看起来没有明显改善，并且生成分子存在复杂、偏疏水、与期望“合理分子”不一致的问题。

围绕这个问题，分析重点放在三件事：

1. 训练过程是否真的没有改善。
2. docking、classifier、分子性质之间是否存在冲突。
3. 当前 `chembl_classifier_selected_v0_1.pt` 在基础 DiffSBDD 生成分子上的输出分布是否适合作为 RL reward。

结论概括：

| 结论 | 判断 |
|---|---|
| 训练是否完全无效 | 不是。有效率、QED、Lipinski、descriptor MMD 有改善 |
| 核心目标是否稳定改善 | 没有。docking、classifier、reward 没有形成一致方向 |
| 当前 classifier 是否适合硬 reward | 不适合。输出双峰且大量样本落到 `-1` 地板 |
| 复杂分子问题 | 明确存在，尤其是非芳香多环、脂肪族复杂结构 |
| 下一步核心 | 重定义正样本，移除 complex-but-valid 正类，使用 soft classifier reward |

## 二、已有相关数据与分析产物

| 类型 | 路径 | 说明 |
|---|---|---|
| 训练输出 | `OUT/7gpu-dock-classifier-v0-1_20260528_000841` | 25 epoch，docking + classifier |
| 训练详细分析 | `OUT/7gpu-dock-classifier-v0-1_20260528_000841/analysis_detailed_properties/` | 训练图表、分子性质、top 分子、相关性 |
| 基础 DiffSBDD 样本 | `OUT/260407_generate_ligands_10000/260407generated_10000_filtered.sdf` | 5829 个 filtered valid molecules |
| classifier 分布分析 | `OUT/260407_generate_ligands_10000/classifier_selected_v0_1_base_distribution/` | classifier probability、reward 映射、性质分箱 |
| 训练分析脚本 | `analysis/analyze_7gpu_dock_classifier_v0_1_20260528.py` | 生成训练详细分析 |
| classifier 分布脚本 | `analysis/evaluate_classifier_on_base_diffsbdd.py` | 评估基础样本 classifier 输出 |

## 三、训练配置

| 项 | 值 |
|---|---|
| run name | `7gpu-dock-classifier-v0-1` |
| GPU | 7 |
| epoch | 25 |
| batch size per GPU | 64 |
| total generated per epoch | 448 |
| reward enabled | `docking`, `chembl_classifier` |
| docking weight | 0.5 |
| classifier weight | 0.5 |
| per-component normalization | enabled |
| total reward normalization | z-score + clamp `[-2, 2]` |
| classifier model | `chembl_classifier_selected_v0_1.pt` |

当前配置中以下 reward 均为 0：

| reward | 状态 |
|---|---|
| QED | disabled |
| SA | disabled |
| ligand efficiency | disabled |
| ChEMBL prior | disabled |
| diversity | disabled |
| LogP distribution | disabled |

这意味着训练主要由 raw docking 与 classifier 两个信号决定，没有直接约束复杂度、合成可及性、LogP 分布或 size-normalized docking。

## 四、训练总体结果

| 指标 | 数值 |
|---|---:|
| epoch | 25 |
| 总生成分子 | 11200 |
| RDKit valid 总数 | 7943 |
| 总体 valid ratio | 70.92% |
| 末轮 valid ratio | 74.11% |
| 最佳 valid ratio | 77.68%，epoch 24 |
| 末轮 avg docking | -7.7229 |
| 最佳 avg docking | -7.9400，epoch 21 |
| 末轮 avg classifier reward | -0.8754 |
| 末轮 reward mean | -0.5658 |
| 末轮 unique SMILES | 332 / 332 |
| 末轮 unique Murcko scaffolds | 301 |

训练并非完全没有变化。分子性质有一定改善：

| 指标 | epoch 1 | epoch 25 | 变化 |
|---|---:|---:|---|
| LogP mean | 0.32 | 1.41 | 上升 |
| TPSA mean | 103.83 | 83.26 | 下降 |
| QED mean | 0.430 | 0.496 | 上升 |
| SA mean | 5.01 | 4.92 | 略好 |
| Lipinski pass | 72.8% | 80.1% | 上升 |
| carbon fraction | 0.655 | 0.706 | 上升 |
| hetero atoms | 7.55 | 6.33 | 下降 |

解释：模型确实发生了分布移动，但核心 reward 目标没有形成稳定一致的提升。

## 五、为什么看起来没有明显改善

### 5.1 classifier reward 饱和

末轮有效分子中：

| classifier reward 条件 | 比例 |
|---|---:|
| reward > 0 | 7.5% |
| reward > 0.5 | 5.7% |
| reward <= -0.5 | 89.5% |
| reward 近似 -1 | 83.7% |

判断：classifier reward 不是连续稠密信号，而是把大部分有效分子压到同一个惩罚地板。PPO 很难从大量相同 `-1` 样本里学习细粒度方向。

### 5.2 docking 与 classifier 目标冲突

末轮不同分子集合对比：

| group | LogP | MW | SA | rings | classifier reward |
|---|---:|---:|---:|---:|---:|
| top reward 10% | 1.58 | 282.5 | 4.14 | 1.68 | 0.51 |
| top docking 10% | 3.74 | 383.7 | 4.97 | 3.76 | -0.88 |
| 全部有效分子 | 1.41 | 326.0 | 4.92 | 2.68 | -0.83 |

判断：top docking 分子更大、更疏水、更复杂，并且大多被 classifier 判差；top reward 分子更合理，但 docking 不够突出。两个 reward 分量没有形成一致方向。

### 5.3 raw docking 均值被异常污染

epoch 23 出现极端正 docking outlier，其中一个 `docking_raw = 1012.356`，导致 `avg_docking` 被拉坏。epoch 24 有 50 个 `docking_raw = 0`。

判断：只看 `avg_docking` 会误判。后续应固定看 docking median、q10、q90、nonzero mean、正值 outlier 数量和 docking failure 数量。

### 5.4 reward mean 不是绝对进步指标

本 run 开启了 per-component normalization 与全局 z-score + clamp。`reward_mean` 是相对归一化后的统计量，不适合作为跨 epoch 的绝对改善指标。

更可靠的判断指标是：

| 类别 | 应看指标 |
|---|---|
| docking | median、q10、q90、nonzero mean |
| classifier | probability 分布、positive ratio、floor ratio |
| 分子性质 | LogP、SA、rings、MW、TPSA、QED、Lipinski |
| 复杂度 | bridgehead、spiro、ring count、SA、rot bonds |
| 失败 | invalid、fragmented、sanitize failed、docking failure |

## 六、复杂分子问题

末轮有效分子中：

| 指标 | 比例 |
|---|---:|
| SA > 5 | 44.9% |
| SA > 6 | 9.6% |
| rings >= 4 | 26.2% |
| rings >= 6 | 3.0% |
| MW > 500 | 5.1% |
| rotatable bonds > 10 | 9.0% |
| Lipinski fail | 19.9% |
| carbon fraction > 0.8 | 21.1% |
| aromatic rings = 0 | 96.7% |

判断：生成分子不是典型芳香药物骨架，而是大量非芳香、多环、脂肪族、桥环或笼状复杂结构。raw docking 容易奖励这种依靠体积和疏水接触获得好分的分子。

当前配置没有直接约束这些问题，因此模型可以通过复杂度和疏水性 exploit docking。

## 七、基础 DiffSBDD 样本上的 classifier 输出

对 `260407generated_10000_filtered.sdf` 中 5829 个有效分子使用 `chembl_classifier_selected_v0_1.pt` 打分：

| 指标 | 数值 |
|---|---:|
| scored molecules | 5829 |
| classifier prob mean | 0.3545 |
| classifier prob median | 0.0240 |
| prob < 0.01 | 45.7% |
| prob < 0.05 | 52.8% |
| prob < 0.20 | 59.0% |
| prob >= 0.50 | 35.2% |
| prob >= 0.80 | 29.8% |
| mapped reward floor = -1 | 59.0% |
| mapped reward positive | 35.2% |

判断：基础模型上 classifier 输出呈双峰分布。一部分样本接近 0，一部分接近 1。它并非完全没有正样本，但当前 hard reward mapping 会让 59% 基础分子直接落到 `-1` 地板，因此不适合作为稠密 RL reward。

## 八、classifier 与“合理分子”并未完全对齐

临时 reasonable region 定义：

| 指标 | 范围 |
|---|---|
| MW | 180-450 |
| LogP | 0.5-3.8 |
| SA | <= 5.0 |
| ring_count | <= 4 |
| rot_bonds | <= 8 |
| heavy_atoms | 12-32 |
| carbon_fraction | <= 0.78 |
| hetero_atoms | >= 2 |
| TPSA | 35-130 |
| formal_charge | -1 到 1 |

分组结果：

| group | n | prob median | prob >= 0.5 | prob < 0.2 |
|---|---:|---:|---:|---:|
| reasonable=True | 939 | 0.9931 | 78.7% | 16.5% |
| reasonable=False | 4890 | 0.0048 | 26.9% | 67.2% |

判断：classifier 确实偏好 reasonable 区域，但 outside reasonable 的分子仍有 26.9% 被判为正。粗略估算，`prob >= 0.5` 的高分分子中只有约 36% 落在 reasonable region。

高概率组仍不够干净：

| prob bin | SA > 5 | rings >= 4 | LogP > 4 | carbon fraction > 0.8 |
|---|---:|---:|---:|---:|
| prob < 0.01 | 76.8% | 78.3% | 39.4% | 30.9% |
| prob >= 0.80 | 13.2% | 39.1% | 25.0% | 14.3% |

结论：classifier 学到了一部分复杂和疏水是不好的，但正类定义仍然过宽，复杂但有效的分子仍可能被 classifier 当作好分子。

## 九、核心问题归纳

| 问题 | 证据 | 影响 |
|---|---|---|
| classifier 正类过宽 | 高概率组仍有较多复杂/疏水分子 | teacher 与真实目标不一致 |
| classifier reward 过硬 | 末轮 83.7% 近似 `-1` | PPO 学不到细粒度方向 |
| docking 与 classifier 冲突 | top docking classifier reward 为 -0.88 | 两个 reward 分量相互拉扯 |
| raw docking 可被复杂度 exploit | top docking MW、LogP、rings 更高 | 生成复杂疏水分子 |
| 缺少复杂度约束 | SA、LE、LogP distribution 均关闭 | 无法限制不合理结构 |

一句话判断：

当前训练的主要问题不是模型完全不学，而是 reward teacher 和目标不一致。模型获得了一些分布层面的改善，但没有被稳定地推向“合理、不过分疏水、不复杂、docking 好”的目标区域。

## 十、下一步建议

### 10.1 重建 classifier 正样本定义

不要再使用随机 ChEMBL 作为正类。正类应定义为：

| label | 定义 |
|---:|---|
| 1.0 | 合理、不过分疏水、不复杂、可合成的 ChEMBL 子集 |
| 0.3-0.5 | 边界 ChEMBL，略复杂但不明显错误 |
| 0.0 | complex/hydrophobic/undesired ChEMBL 与生成 hard negatives |

复杂但有效的 ChEMBL 分子应从正类移除，并转为 hard negative 或低标签边界样本。

### 10.2 加入当前生成的 hard negatives

应从当前 run 中抽取：

| hard negative 来源 | 条件 |
|---|---|
| top docking 复杂分子 | top docking 10% 且 SA 高 |
| 疏水复杂分子 | LogP 高、carbon fraction 高 |
| 多环复杂分子 | rings 多、bridgehead/spiro 多 |
| 规则失败分子 | Lipinski fail、SA > 6、MW > 500 |

目的：明确告诉 classifier，docking 高但复杂疏水不是目标。

### 10.3 classifier reward 改为 soft signal

应记录并使用：

| 字段 | 用途 |
|---|---|
| classifier probability | 判断原始分布 |
| classifier logit | 做连续 reward |
| mapped soft reward | RL 使用 |
| floor ratio | 检查是否再次饱和 |

避免多数分子被映射成同一个 `-1`。

### 10.4 reward 改为受约束 docking

真实目标应写成：

```text
在 LogP 和复杂度受控条件下提高 docking
```

建议形式：

```text
effective_docking = docking_quality * logp_gate * simplicity_gate
```

| 组件 | 作用 |
|---|---|
| docking_quality | raw docking + ligand efficiency |
| logp_gate | LogP 在合理窗口内不打折，过疏水则打折 |
| simplicity_gate | SA、rings、heavy atoms、rot bonds 过高则打折 |

这样复杂疏水分子即使 raw docking 好，也不会拿到高 reward。

### 10.5 下一版训练建议

先做一版干净实验：

| reward | 建议 |
|---|---|
| docking / ligand efficiency | 打开 |
| LogP window | 打开 |
| SA / simplicity | 打开 |
| classifier | 先关闭或极低权重 |

目标是先验证模型能否在不过分疏水、不复杂的情况下提高 docking。之后再接入重训后的 desired classifier。

## 十一、后续必须监控的指标

| 类别 | 指标 |
|---|---|
| classifier | prob mean、median、q10、q90、p < 0.2、p >= 0.5、floor ratio |
| docking | mean、median、q10、q90、nonzero mean、positive outliers |
| complexity | SA mean、SA > 5、SA > 6、rings >= 4、rings >= 6 |
| hydrophobicity | LogP mean、LogP > 4、LogP > 5、carbon fraction |
| drug-likeness | QED、Lipinski pass、TPSA、HBA、HBD |
| size | MW、heavy atoms、rot bonds |
| top groups | top docking 10%、top reward 10%、top classifier 10% 的性质 |

只看 `reward_mean` 和 `avg_docking` 不足以判断训练是否朝正确方向走。

## 十二、命令与验证

已运行：

```bash
conda run -n dif python -m py_compile analysis/analyze_7gpu_dock_classifier_v0_1_20260528.py
conda run -n dif python analysis/analyze_7gpu_dock_classifier_v0_1_20260528.py
conda run -n dif python -m py_compile analysis/evaluate_classifier_on_base_diffsbdd.py
conda run -n dif python analysis/evaluate_classifier_on_base_diffsbdd.py
```

图表已生成并通过 `file` 抽查为有效 PNG。

