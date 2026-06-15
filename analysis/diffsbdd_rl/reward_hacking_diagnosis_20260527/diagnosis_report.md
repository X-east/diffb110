# DiffB110 Reward Hacking 与正负样本策略诊断

生成时间：2026-05-27 11:55:16

## 一、分析范围

- 分子级记录：267,128 rows
- 覆盖 run 数：57
- 唯一 SMILES/异常值：170,061
- 输入目录：`OUT/*/epochs/epoch_*/molecules.csv`
- 本分析只读训练输出、ChEMBL property 表和当前分类器负样本文件；没有生成训练集。

## 二、核心结论

1. 当前问题不是单个坏结构，而是 RL 后生成分布整体偏移。
2. 全局有效分子里，严格药物合理核心通过率只有 0.280。
3. 三元环比例极高：全局 valid 分子中 0.519 含三元环。
4. 高碳问题显著：全局 valid 分子中 C>=0.90 比例为 0.175，而 ChEMBL36 baseline 只有 0.015。
5. 低杂原子问题显著：全局 valid 分子中 hetero<=1 比例为 0.108，而 ChEMBL36 baseline 只有 0.004。
6. 高 reward 区域仍富集坏结构，说明 reward 可以被利用，不能只靠当前二分类器兜底。

## 三、最关键的崩坏证据

### 7gpu-dock-ecfp4-fcfp4_20260525_212140

- epoch 1：C>=0.90 为 0.012，strict core 为 0.276。
- epoch 34 reward 最好：C>=0.90 升到 0.357，三元环 0.578，复杂环 0.399。
- epoch 62：C>=0.90 为 0.435，hetero<=1 为 0.272，三元环 0.682，复杂环 0.609，strict core 降到 0.172。
- 解释：docking + ChEMBL prior + logP distribution 能维持有效率，但会推动高碳、多环、少杂原子结构。

### 6gpu-chembl-only_20260521_154957

- epoch 16 reward 最好：C>=0.90 为 0.643，hetero<=1 为 0.464，no hetero 为 0.200。
- epoch 20：C>=0.90 为 0.720，hetero<=1 为 0.548，strict core 只有 0.152。
- 解释：仅 ChEMBL prior/指纹相似度不足以约束元素组成，模型会找到高碳相似片段捷径。

### 7gpu-classifier-only_20260526_210715

- epoch 23 reward 最好，但三元环为 0.544，复杂环为 0.340。
- epoch 26 开始有效率低于 0.5，epoch 64 有效率为 0。
- 解释：当前分类器奖励单独使用时会被策略推到不可采样/不可解析区域，不能作为唯一 reward。

## 四、ChEMBL 正样本池

- ChEMBL36 cleaned 总量：1,550,090
- property_core_loose：1,439,346
- property_core_lipinski0：1,355,416
- recommended_strict_property_pool：1,280,455

结论：正样本不缺，问题是不能随机抽 100k。应从 strict property pool 中 scaffold-stratified + descriptor-stratified 抽样。

## 五、当前负样本覆盖缺口

当前负样本标签分布：

{
  "prop_all_carbon": 7967,
  "pain2_pseudo_normal": 4369,
  "prop_too_complex": 2231,
  "pain1_sulfur": 1501,
  "prop_too_hydrophilic": 1467,
  "prop_sulfur_excess": 362,
  "pain4_extreme_outlier": 236,
  "pain3_dock_collapse": 110
}

主要缺口：

- 三元环没有作为独立负样本标签。
- 复杂桥环/稠合非芳香环没有作为独立负样本标签。
- 高 reward 但坏结构的样本没有系统性 hard-negative mining。
- 负样本以 all-carbon 和若干 pain point 为主，对“看起来单分子合理但分布错误”的模式覆盖不足。

## 六、建议的数据集方向

正样本：

- 从 recommended strict ChEMBL pool 抽样。
- 按 Bemis-Murcko scaffold 限额，避免大系列支配。
- 按 heavy atoms、carbon fraction、hetero count、ring count、logP、TPSA 分箱抽样。
- 不要把 ChEMBL 中高碳、低杂原子、极端 logP/TPSA、复杂稠环样本打成强正样本。

负样本：

- 从 OUT 中按 failure category 分层收集。
- 必须新增：three_member_ring、ring_complex、high_carbon_90、low_hetero、few_element_types、high_reward_bad、invalid_after_rl。
- hard negative 应优先来自 reward 高但 strict_core_pass=0 的分子。
- 不要把低 ChEMBL similarity 单独当负样本；低相似度只有叠加坏结构/坏性质时才是负样本。

奖励上线：

- 分类器只能作为 soft distribution reward。
- 三元环、高碳、无/低杂原子、复杂环应作为显式 hard penalty 或 hard filter。
- 每轮 RL 后做 hard-negative mining：高 classifier/reward 且规则失败的分子加入下一版负样本。

## 七、产物说明

- `molecule_flags_compact.csv.gz`：每个生成分子的结构标签和关键描述子。
- `run_summary.csv`：每个 run 的整体统计。
- `major_run_summary.csv`：主要非 smoke run 的统计。
- `selected_epoch_summary.csv`：关键 run 的逐 epoch 统计。
- `top_reward_decile_summary.csv`：每个 run top reward 10% 的坏结构比例。
- `top_reward_bad_examples.csv`：高 reward 坏分子代表样例。
- `chembl_baseline.json`：ChEMBL36 baseline。
- `chembl_positive_pool.json`：推荐正样本池规模估计。
- `negative_label_counts.json`：当前负样本标签统计。
- `key_config_summary.json`：关键 run 的 reward/train 配置摘要。
- `selection_criteria_rationale.md`：筛选标准的简版依据说明。
- `screening_criteria_evidence_matrix.md`：逐项阈值、ChEMBL 基线、OUT 崩坏证据和答辩口径。
