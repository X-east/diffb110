# 260407 generated SDF 的 ChEMBL 判别器测试

## 一、动机/背景

在直接开启 7 卡 `docking:0.5 + chembl_classifier:0.5` 强化学习前，需要先确认新训练的 v0.1 ChEMBL 判别器在一个真实生成集合上是否能识别已知坏模式。这个集合来自 `OUT/260407_generate_ligands_10000/260407generated_10000_filtered.sdf`，代表模型历史生成分子的实际分布，比只看分类器 held-out test 更能暴露 reward hacking 漏洞。

## 二、输入与输出

- 输入 SDF: `OUT/260407_generate_ligands_10000/260407generated_10000_filtered.sdf`
- 判别器: `chembl_classifier/models/chembl_classifier_selected_v0_1.pt`
- 特征器: 与训练奖励一致的 `morgan_count_rdkit_2d`
- 结果目录: `analysis/classifier_test_260407_sdf_20260527/`
- 分子级 CSV: `260407generated_10000_filtered_classifier_scores.csv.gz`
- 总结 JSON: `summary.json`
- 分组风险 JSON: `subgroup_risk_summary.json`
- 图:
  - `classifier_probability_hist.png`
  - `risk_flag_rates.png`
  - `probability_vs_carbon_fraction.png`

## 三、整体结果

虽然文件名含 `10000`，RDKit 从该 filtered SDF 中实际读到 5829 条记录，其中 5808 条可以完成特征化和判别器打分，有效率为 99.64%。

判别器输出的 `P(ChEMBL-like)` 分布：

| 指标 | 数值 |
|---|---:|
| mean | 0.3545 |
| median | 0.0240 |
| p05 | 6.67e-10 |
| p25 | 6.73e-05 |
| p75 | 0.9364 |
| p95 | 0.99996 |
| P >= 0.5 | 35.35% |
| P >= 0.8 | 29.92% |
| P < 0.2 | 59.21% |

按当前训练奖励映射后，平均 mapped reward 为 -0.2928，中位数为 -1.0。这说明判别器会强烈惩罚该历史生成集合中的大部分分子，并不是无差别放过。

## 四、坏模式识别情况

全体有效分子的风险比例：

| 风险项 | 全体比例 | 该风险项平均 P(ChEMBL-like) |
|---|---:|---:|
| C >= 0.90 | 2.70% | 0.0294 |
| hetero <= 1 | 1.02% | 0.0059 |
| CO-only mostly carbon | 2.08% | 0.0530 |
| three-member ring | 36.40% | 0.1391 |
| ring_complex | 64.94% | 0.2617 |
| has benzene-like ring | 48.09% | 0.4486 |

高分组 `P >= 0.5` 中的风险比例：

| 风险项 | 高分组比例 |
|---|---:|
| C >= 0.90 | 0.19% |
| hetero <= 1 | 0.00% |
| CO-only mostly carbon | 0.29% |
| three-member ring | 13.74% |
| ring_complex | 47.59% |
| has benzene-like ring | 61.57% |

低分组 `P < 0.2` 中的风险比例：

| 风险项 | 低分组比例 |
|---|---:|
| C >= 0.90 | 4.42% |
| hetero <= 1 | 1.72% |
| CO-only mostly carbon | 3.26% |
| three-member ring | 50.92% |
| ring_complex | 76.21% |
| has benzene-like ring | 39.92% |

## 五、结论

这个判别器对你最担心的“碳元素过量、杂原子极少、只有 C/O 且偏碳”的方向是有效的：这些分子在全体中已有较低概率，在高分组中几乎被清掉。

但它对“复杂环/稠环”和一部分“三元环”仍然有漏网。尤其是 `P >= 0.5` 的分子里，`ring_complex` 仍有 47.59%，`three_member_ring` 仍有 13.74%。这意味着当前 `docking + classifier` 大跑可以作为第一轮实验，但后续结果分析必须重点盯住环系统分布；如果 RL 仍向复杂环聚集，下一版负样本或奖励约束需要更强地覆盖 ring topology。

## 六、7 卡大测试启动记录

已启动配置 `launch/7gpu_dock_classifier_v0_1.yaml`：

- run dir: `OUT/7gpu-dock-classifier-v0-1_20260527_191616`
- 启动日志: `analysis/classifier_test_260407_sdf_20260527/7gpu_dock_classifier_launch.log`
- rank-0 训练日志: `OUT/7gpu-dock-classifier-v0-1_20260527_191616/logs/rank_00/logs/training_20260527_191618.log`
- GPU: 使用 0-6；GPU 7 仍为 `pmemd.cuda` 占用，未被本次训练使用。

第 1 个 epoch 已完成并写入 `training_summary.csv`。日志确认启用奖励组件只有 `docking` 和 `chembl_classifier`，权重均为 0.5。

注意：`training_summary.csv` 中旧列 `avg_chembl_reward` 第 1 个 epoch 为 0，但日志显示动态字段注册为 `chembl_classifier_reward`。这更像是 summary 旧列名没有映射到新判别器字段，而不是判别器未参与奖励；后续分析不要只依赖这个旧列判断分类器是否生效。
