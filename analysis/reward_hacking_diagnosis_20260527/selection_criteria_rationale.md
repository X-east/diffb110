# 正负样本筛选标准依据说明

生成时间：2026-05-27

详细证据矩阵见：
`analysis/reward_hacking_diagnosis_20260527/screening_criteria_evidence_matrix.md`

## 一、总原则

本项目的筛选标准不是要证明“不满足者一定不是药物”，而是要为
DiffSBDD + DDPO 的强化学习过程建立一个保守、可解释、抗 reward hacking 的
分布约束。

因此，每条标准来自三类证据之一：

1. **文献规则**：Lipinski、Veber、QED、PAINS/MCF 等成熟药物化学经验。
2. **ChEMBL36 经验分布**：以本项目清洗后的 1,550,090 条 ChEMBL36 分子为
   empirical baseline。
3. **本项目 RL 崩坏证据**：从 `OUT/*/epochs/*/molecules.csv` 的 267,128 条
   生成记录中统计 reward hacking 发生的结构富集模式。

筛选标准应被称为：

> conservative drug-like / ChEMBL-like core for reward-hacking control

而不是“所有药物必须满足的绝对规则”。

## 二、各项标准的依据

| 标准 | 推荐用途 | 阈值 | 依据 |
|---|---:|---:|---|
| MW | 正样本核心、软过滤 | 150-500 | Lipinski Rule of Five 使用 MW > 500 作为口服吸收/通透性风险信号；本项目 ChEMBL36 预处理也使用 150-500。 |
| heavy atoms | 正样本核心、模型支持范围 | 10-30 | 本项目 ChEMBL36 清洗已使用 10-30；也避免 fragment-like 和过大结构。 |
| logP | 正样本核心、reward 监控 | diagnostic: -1 到 6；strict: 0 到 5.5 | Lipinski 使用 CLogP > 5 作为风险信号；本项目 ChEMBL p95=5.33、p99=6.32，故 5.5 是保守上界。 |
| TPSA | 正样本核心、reward 监控 | diagnostic: 20-140；strict: 20-120 | Veber 等提出 TPSA <= 140 与较好口服生物利用度相关；20 下界来自本项目 ChEMBL baseline，TPSA <20 仅 1.9%，用于防止低极性高碳漂移。 |
| HBD/HBA | 正样本核心 | HBD <=5, HBA 1-10 | Lipinski Rule of Five；HBA >=1 是为了避免无极性/全碳结构。 |
| rotatable bonds | 正样本核心 | <=10 | Veber 等提出 rotatable bonds <=10 与较好 oral bioavailability 相关。 |
| carbon fraction | 正样本核心、负样本标签 | positive <=0.88；warning >=0.90；hard negative >=0.95 | 不是经典文献规则，而是 ChEMBL36 empirical rule：C>=0.90 仅 1.5%，C>=0.95 仅 0.2%；生成分子全局 C>=0.90 为 17.5%，部分 run 达 72.0%。 |
| hetero atoms | 正样本核心、负样本标签 | positive >=2；warning <=1；hard negative =0 | ChEMBL36 中 hetero<=1 只有 0.4%；生成分子全局 hetero<=1 为 10.8%，部分 run 达 54.8%。 |
| element diversity | 负样本标签、分布监控 | element_types <=2 作为 warning | 当前生成分子中 few_element_types 全局为 39.9%，说明模型偏向 C/H/O 或 C/H 等低信息元素组合。 |
| three-member ring | 负样本标签、hard penalty 候选 | any ring size == 3 | 单个 cyclopropane/epoxide 不必绝对排除，但本项目生成分子中三元环比例全局 51.9%，关键 run 可达 68.2%，这是分布崩坏而非正常药物样多样性。 |
| ring count / ring complexity | 负样本标签、分布监控 | ring_count >=5 或 max fused cluster >=4 或 bridgehead/spiro >=2 | ChEMBL36 rings>=6 只有 0.9%；生成分子 ring_complex 全局 24.9%，关键 run 可达 60.9%。复杂度过高也与 QED 中“too complex”化学吸引力下降一致。 |
| aromatic ring count / fused aromatic systems | 正样本上限、负样本标签 | aromatic_rings <=4；large fused aromatic cluster 作为负样本 | Ritchie & Macdonald 讨论过 aromatic ring count 对 developability 的负面影响；PAH-like/大稠环也常与低溶解性、高疏水性、flatness 和 assay/liability 风险相关。 |
| PAINS / Brenk / MCF | 正样本排除、负样本标签 | 出现 alert 则不作为强正样本 | Baell & Holloway 提出 PAINS 过滤；Brenk 等提出结构警报用于筛除不良/反应性官能团。注意结构警报应与整体性质结合使用，不宜单独决定一切。 |
| strict_core_pass | 诊断指标，不是唯一标签 | 满足 property + composition + ring sanity | 用于判断 RL 分布是否仍位于保守药物样核心区。当前全局通过率只有 28.0%，说明需要加入分布约束。 |

## 三、为什么 ChEMBL 里大多数满足，而生成分子不满足

本项目 ChEMBL36 已经经过预清洗：

- 输入：2,854,815 条
- 去盐/parent/标准化/去重后保留：1,550,090 条
- 过滤条件：MW 150-500、heavy atoms 10-30、formal charge = 0

在这个 cleaned ChEMBL36 上：

- C>=0.90：1.5%
- C>=0.95：0.2%
- hetero<=1：0.4%
- rings>=6：0.9%
- logP>5.5：3.9%
- TPSA<20：1.9%
- recommended strict property pool：1,280,455 条，占 82.6%

而生成分子全局 valid 集合中：

- strict_core_pass：28.0%
- C>=0.90：17.5%
- C>=0.95：6.9%
- hetero<=1：10.8%
- three-member ring：51.9%
- ring_complex：24.9%
- logP>5.5：12.2%
- TPSA<20：12.6%

因此，这些阈值的核心理由不是“人为挑剔”，而是：

> 生成分子在这些维度上相对 ChEMBL36 baseline 出现了数量级级别的富集。

## 四、如何在论文/汇报中表述

推荐表述：

> We define a conservative drug-like core by combining established oral
> drug-likeness heuristics, empirical quantiles from the cleaned ChEMBL36
> reference set, and failure modes observed in DDPO-optimized samples. The
> criteria are not intended to reject all molecules outside the region, but to
> identify distributional drift and reward-hacking modes during reinforcement
> learning.

中文表述：

> 本研究中的 strict drug-like core 不是药物可成药性的绝对判据，而是面向
> 强化学习分子生成的保守分布约束。该约束同时结合了 Lipinski/Veber 等
> 药物化学经验规则、清洗后 ChEMBL36 的经验分布，以及 DDPO 训练后实际出现的
> reward hacking 模式，用于识别和抑制模型向高碳、低杂原子、三元环和复杂环
> 结构的异常漂移。

## 五、正负样本标签建议

### 正样本

正样本不应是随机 ChEMBL 100k，而应从 strict ChEMBL property pool 中抽样：

- scaffold-stratified
- descriptor-stratified
- 限制同一 scaffold 过度重复
- 不把高碳、低杂原子、极端 logP/TPSA、复杂稠环分子作为强正样本

### 负样本

负样本应按失败类别分层，而不是只依赖 all-carbon 或少数 pain point：

- high_carbon_90
- high_carbon_95
- no_hetero
- low_hetero
- few_element_types
- three_member_ring
- ring_complex
- high_reward_bad
- invalid_after_rl
- sulfur_excess
- PAINS/Brenk/MCF alert

其中 `high_reward_bad` 最重要，定义为：

```text
reward 高 / classifier 高
AND strict_core_pass = 0
AND 至少命中一个结构或性质风险标签
```

这类样本直接对应 reward hacking 漏洞。

## 六、主要参考文献和来源

- Lipinski et al. Experimental and computational approaches to estimate
  solubility and permeability in drug discovery and development settings.
  Adv Drug Deliv Rev. 1997/2001. https://pubmed.ncbi.nlm.nih.gov/11259830/
- Veber et al. Molecular properties that influence the oral bioavailability of
  drug candidates. J Med Chem. 2002. https://pubmed.ncbi.nlm.nih.gov/12036371/
- Bickerton et al. Quantifying the chemical beauty of drugs. Nature Chemistry.
  2012. https://pmc.ncbi.nlm.nih.gov/articles/PMC3524573/
- Baell and Holloway. New substructure filters for removal of pan assay
  interference compounds (PAINS) from screening libraries and for their
  exclusion in bioassays. J Med Chem. 2010.
  https://pubs.acs.org/doi/abs/10.1021/jm901137j
- Ritchie and Macdonald. The impact of aromatic ring count on compound
  developability: are too many aromatic rings a liability in drug design?
  Drug Discovery Today. 2009.
- GuacaMol benchmark. https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839
- MOSES benchmark. https://arxiv.org/abs/1811.12823
