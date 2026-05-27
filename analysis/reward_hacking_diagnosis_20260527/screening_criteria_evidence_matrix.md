# 正负样本筛选标准证据矩阵

生成时间：2026-05-27

## 结论

本项目的筛选标准应定义为：

> 用于 DiffSBDD + DDPO reward-hacking control 的 conservative
> drug-like / ChEMBL-like distribution guardrail。

它不是“药物/非药物”的绝对判定，也不是要否定所有超出阈值的真实活性分子。
它的目的更具体：在强化学习奖励模型中，让分类器高分区域稳定落在
ChEMBL36 的主分布内，并显式惩罚本项目已经观察到的高碳、少杂原子、三元环、
复杂环和高 reward 坏结构捷径。

因此，每个阈值必须同时回答三个问题：

1. 文献或药物化学经验上为什么相关。
2. 在清洗后的 ChEMBL36 中是不是主分布或罕见尾部。
3. 在本项目 OUT 训练结果中是不是 reward hacking 富集方向。

只有同时满足第 2 点和第 3 点的标准，才适合作为本项目的强负样本或硬惩罚；
只满足文献经验但本项目没有崩坏证据的标准，更适合作为软监控指标。

## 标签原则

分类器训练集不要做成简单的“ChEMBL = 1，非 ChEMBL = 0”。更稳妥的标签定义是：

| 标签层级 | 含义 | 用途 |
|---|---|---|
| 强正样本 | 位于 ChEMBL36 主体性质分布内、结构警报少、scaffold 多样的分子 | 教分类器什么是稳定的 ChEMBL-like / drug-like core |
| 灰区样本 | ChEMBL 中真实存在但位于尾部的分子，或单个性质略越界但无明显结构问题的分子 | 不作为强负样本；可用于校准或留出集 |
| 强负样本 | OUT 中生成的失败模式、已知结构警报、高 reward 但 strict_core_fail 的样本 | 堵住 DDPO 可利用的 reward 漏洞 |

这样做的理由是：强化学习会主动搜索分类器的盲区。相比普通二分类任务，
这里更怕“坏分子被打高分”，而不是“少数边缘好分子被打低分”。所以强正样本
要高精度，强负样本要覆盖真实崩坏模式，灰区不要硬塞进 0/1。

## 本地数据基线

本次分析覆盖：

- OUT 生成记录：267,128 rows，57 个 runs。
- 清洗后 ChEMBL36：1,550,090 molecules。
- 推荐 strict ChEMBL property pool：1,280,455 molecules，占 cleaned ChEMBL36 的 82.6%。

核心对比：

| 指标 | ChEMBL36 cleaned | OUT valid generated | 富集倍数/解释 |
|---|---:|---:|---|
| C >= 0.90 | 1.46% | 17.5% | 约 12.0 倍，明确高碳漂移 |
| C >= 0.95 | 0.17% | 6.9% | 约 40.0 倍，接近烃类/极低极性捷径 |
| hetero <= 1 | 0.40% | 10.8% | 约 26.8 倍，少杂原子崩坏 |
| logP > 5.5 | 3.89% | 12.2% | 约 3.1 倍，疏水性偏移 |
| TPSA < 20 | 1.86% | 12.6% | 约 6.8 倍，极低极性偏移 |
| rings >= 6 / ring_complex | 0.86% rings>=6 | 24.9% ring_complex | 指标不完全等价，但方向一致：多环/复杂环明显富集 |
| three-member ring | 未在 ChEMBL baseline 中统计 | 51.9% | 单个三元环不是绝对错误，但 51.9% 是生成分布崩坏 |
| strict_core_pass | property pool 82.6% | 28.0% | 生成分布大幅偏离 conservative core |

这张表是最重要的答辩依据：筛选标准不是主观偏好，而是针对模型实际崩坏方向的
分布校正。

## 标准逐项依据

| 维度 | 推荐阈值/用法 | 理由 |
|---|---|---|
| heavy atoms | 强正样本 10-30；超出作为灰区或负样本候选 | 与本项目 ChEMBL 清洗范围一致，避免 fragment-like 分子和过大结构进入主奖励分布。recommended pool 中 heavy atoms p95/p99 都是 30，说明 30 是当前训练域上界。 |
| MW | 强正样本 150-500 | Lipinski Rule of Five 将 MW > 500 作为口服吸收/通透性风险信号之一；150 下界用于排除过小 fragment。 |
| logP | 强正样本建议 <=5.0 或 <=5.5；>5.5 作为风险标签 | Lipinski 使用 CLogP > 5 作为风险信号。ChEMBL36 中 logP>5.5 只有 3.89%，OUT valid 中为 12.2%。recommended pool 的 logP p99 为 4.91，所以强正样本不需要包含高 logP 尾部。 |
| TPSA | 强正样本 20-120 或 20-140；TPSA<20 作为风险标签 | Veber 等提出 TPSA <=140 与较好 oral bioavailability 相关。这里额外设 20 下界不是经典口服规则，而是本项目的 anti-hydrophobic-collapse 规则：ChEMBL36 TPSA<20 只有 1.86%，OUT valid 为 12.6%。recommended pool TPSA p01 为 24.39。 |
| HBD/HBA | HBD <=5，HBA 1-10；强正样本可要求 HBA>=1 | 来自 Lipinski Rule of Five。HBA>=1 的项目意义是避免无极性、全碳或近全碳结构被分类器认为是强正样本。 |
| rotatable bonds | <=10 | Veber 等提出 rotatable bonds <=10 与 oral bioavailability 相关。recommended pool 的 rotB p95=8、p99=9，说明 <=10 对 ChEMBL 主分布并不苛刻。 |
| carbon fraction | 强正样本建议 <=0.88；C>=0.90 warning；C>=0.95 hard negative 候选 | 这不是传统药物化学通用规则，而是本项目最强的 empirical rule。cleaned ChEMBL36 C>=0.90 仅 1.46%，C>=0.95 仅 0.17%；OUT valid 分别为 17.5% 和 6.9%。关键 run 可到 C>=0.90 72.0%。 |
| hetero atom count | 强正样本建议 >=2；strict_core 诊断可放宽到 >=1；hetero<=1 作为 hard negative 候选 | cleaned ChEMBL36 hetero<=1 只有 0.40%，hetero<=2 也只有 2.38%；recommended pool 中 hetero p01=3。OUT valid hetero<=1 为 10.8%，关键 run 可到 54.8%。 |
| element diversity | element types <=2 作为风险标签，不单独决定强负 | OUT valid 中 few_element_types 过高，说明模型偏向低信息元素组合。但 C/N/O 的简单分子也可能合理，所以它应作为组合标签，与高碳、低 TPSA、低杂原子一起使用。 |
| three-member ring | 不建议作为强正样本核心；生成样本中出现即标记，尤其 high_reward_bad 中作为 hard negative | cyclopropane、epoxide、aziridine 等三元环在药物中可以存在，因此不能说“有三元环就不是药”。但 OUT valid 中 51.9% 含三元环，关键 run 可达 68.2%，这已经是分布崩坏，不是正常药物化学多样性。 |
| ring count / ring complexity | 强正样本 ring_count <=5 或 <=6；ring_count>=5、max fused cluster>=4、bridgehead/spiro>=2 作为风险标签 | cleaned ChEMBL36 rings>=6 只有 0.86%；recommended pool num_rings p95/p99 都是 5。OUT ring_complex 为 24.9%，关键 run 可达 60.9%。复杂环过多还会增加合成可行性和 developability 风险。 |
| aromatic ring count / fused aromatic system | 强正样本 aromatic_rings <=4；large fused aromatic cluster 作为负样本候选 | Ritchie & Macdonald 讨论过 aromatic ring count 增加与 developability 风险相关。recommended pool aromatic_rings p95/p99 都是 4。大稠环/PAH-like 结构也常伴随平面性、疏水性、低溶解性和 assay liability。 |
| PAINS / Brenk / MCF | alert 分子不作为强正样本；可作为负样本类别之一 | PAINS、Brenk、MCF 是生成分子基准和筛选库常用结构警报。注意 alert 不是绝对死亡判定，应结合性质和任务上下文使用。 |
| high_reward_bad | 必须作为 hard-negative mining 核心类别 | 定义为 reward/classifier 高，但 strict_core_pass=0 且命中至少一个失败标签。这类分子直接证明分类器或组合 reward 存在可利用漏洞，是 DDPO 场景中价值最高的负样本。 |

## 为什么不能只用随机 ChEMBL 正样本

随机抽 ChEMBL 100k 的问题不是样本量不够，而是标签语义不够清楚：

1. ChEMBL 是活性化合物库，不等于所有条目都适合作为“强化学习奖励高分模板”。
2. ChEMBL 内部有大系列、专利系列和相似 scaffold，随机抽样会让常见 scaffold 主导分类器。
3. 随机正样本会混入性质尾部分子，使分类器把高碳、低极性或复杂环尾部也学习成正类。
4. DDPO 会放大分类器边界上的漏洞；普通监督学习中影响不大的假阳性，在 RL 中会变成主要生成模式。

因此，最佳正样本应该从 strict ChEMBL property pool 中做 scaffold-stratified 和
descriptor-stratified 抽样。灰区 ChEMBL 分子可以保留作校准/外部测试，不要当作强负。

## 为什么负样本必须来自 OUT hard negatives

普通结构警报只能告诉模型“传统上哪些结构可疑”，但不能覆盖你自己的扩散模型会犯的错。
你的 OUT 结果已经显示，模型的主要失败模式是：

- 复杂分子和稠环分子。
- 三元环大量富集。
- 碳比例过高。
- 杂原子数量过少或种类单一。
- 单个分子性质看似可接受，但整批生成分布明显不对。
- 高 reward 区域仍包含上述坏结构。

所以负样本必须分层覆盖：

| 类别 | 来源 | 目的 |
|---|---|---|
| high_carbon_90 / high_carbon_95 | OUT + ChEMBL 尾部对照 | 阻止碳元素过量捷径 |
| no_hetero / low_hetero | OUT | 阻止无极性/低极性烃类崩坏 |
| few_element_types | OUT | 阻止元素组成单调化 |
| three_member_ring | OUT | 阻止三元环分布崩坏 |
| ring_complex | OUT | 阻止稠环、桥环、复杂环奖励漏洞 |
| logp_high / tpsa_low | OUT + ChEMBL baseline | 阻止疏水 collapse |
| PAINS/Brenk/MCF | RDKit/规则库 | 阻止传统结构警报 |
| high_reward_bad | OUT top reward decile | 直接修补当前 reward 漏洞 |
| invalid_after_rl | OUT invalid/parse fail | 降低策略走向不可解析空间的倾向 |

其中 high_reward_bad 最重要，因为它直接对应“分类器会被哪些样本骗过”。

## 生成分子不能整体作为负样本

OUT 生成分子默认应视为 unlabeled generated pool，而不是负样本池。原因是：

1. 生成分子中可能已经有真正好的 RE-CmeB 候选分子。
2. 如果把所有生成分子打成 0，分类器会学到“generatedness”而不是
   “drug-likeness”，即把生成器特有的分布痕迹当成负类特征。
3. RL 使用这个分类器后，会被迫回到 ChEMBL 记忆区域，抑制新 scaffold 和
   target-specific 优化。
4. 这会把 reward 从“生成更合理的分子”变成“生成更像训练集来源的分子”，
   属于另一种 reward hacking。

因此，生成分子应分成四类处理：

| 类别 | 标签策略 | 说明 |
|---|---|---|
| generated-good candidate | 不作为负样本；可进入候选正样本/人工复核/外部验证集 | valid、strict_core_pass、无明显结构警报、docking/interaction 合理、novel/scaffold 有价值 |
| generated-gray | 暂不打标签 | 单个指标越界但整体结构可能合理，避免误杀真实候选 |
| generated-hard-negative | 作为负样本 | 命中明确失败模式，尤其 high_reward_bad、C>=0.95、no hetero、ring_complex、三元环异常富集等 |
| invalid/generated-fail | 可作为有效性或解析失败负样本 | RDKit parse/sanitize fail、无法可靠计算性质或结构明显不成立 |

最安全的做法是：OUT 分子默认不参与二分类标签；只有满足明确失败定义的样本才进入负样本。
相反，满足 strict_core 且有目标蛋白 docking/interaction 支持的生成分子，可以单独进入
candidate pool，而不是混进普通 ChEMBL 正样本。

## 推荐答辩口径

可以这样表述：

> We did not use a single arbitrary drug-likeness cutoff. Instead, we defined
> a conservative ChEMBL-like core by combining established medicinal chemistry
> heuristics, empirical distributions from the cleaned ChEMBL36 reference set,
> and failure modes observed in DDPO-optimized molecules. Molecules outside
> this region were not automatically treated as non-drugs; they were used as
> hard negatives only when they matched observed reward-hacking patterns, such
> as high carbon fraction, low heteroatom count, excessive three-member rings,
> or complex fused ring systems.

中文：

> 本研究中的筛选标准不是药物可成药性的绝对判据，而是强化学习分子生成中的
> 保守分布护栏。我们将 Lipinski/Veber/QED/结构警报等已有药物化学经验，
> 清洗后 ChEMBL36 的经验分布，以及 DDPO 训练中实际出现的 reward hacking
> 模式结合起来定义正负样本。只有当某一结构或性质同时位于 ChEMBL 罕见尾部、
> 又在生成结果中显著富集时，才被作为强负样本或硬惩罚候选。

## 与已有生成模型/基准的关系

已有分子生成基准通常不会直接使用“原始数据库全部分子”作为无差别训练真值，
而是先做标准化、清洗、过滤和评估指标设计：

- GuacaMol 使用基于 ChEMBL 的标准化数据集，并用 validity、uniqueness、
  novelty、KL divergence、FCD 等指标评估生成分布，而不只看单分子打分。
- MOSES 使用标准化分子集合和过滤指标，强调生成分布质量、novelty、diversity、
  fragment/scaffold similarity，而不是只优化一个单点 reward。
- PAINS、Brenk/MCF、QED、Lipinski/Veber 等规则常被作为分子生成评价或筛选
  的组成部分，但现代做法通常把它们作为多指标约束，而不是单一生死判据。

这给本项目的启发是：分类器 reward 应该学习 ChEMBL-like 主分布，同时配合显式
failure-category recall、top-reward bad-rate、分布漂移监控，而不是只报告普通
validation accuracy/AUC。

## 参考来源

- Lipinski et al. Experimental and computational approaches to estimate
  solubility and permeability in drug discovery and development settings.
  https://pubmed.ncbi.nlm.nih.gov/11259830/
- Veber et al. Molecular properties that influence the oral bioavailability of
  drug candidates. https://pubmed.ncbi.nlm.nih.gov/12036371/
- Bickerton et al. Quantifying the chemical beauty of drugs.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3524573/
- Baell and Holloway. New substructure filters for removal of pan assay
  interference compounds. https://pubs.acs.org/doi/abs/10.1021/jm901137j
- Ritchie and Macdonald. The impact of aromatic ring count on compound
  developability. https://doi.org/10.1016/j.drudis.2009.07.014
- Brenk et al. Lessons learnt from assembling screening libraries for drug
  discovery. https://doi.org/10.1002/cmdc.200700139
- GuacaMol benchmark. https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839
- MOSES benchmark. https://arxiv.org/abs/1811.12823
