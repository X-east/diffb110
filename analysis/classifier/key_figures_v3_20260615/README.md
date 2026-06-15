# classifier v3 / v0.3.1 关键图合集

这个目录只是把重要图单独复制出来，原始结果仍保留在各自分析目录中。

## 版本口径

- `v0.3`：指 2026-06-10 做的 scaffold / 坏分子规则整理与人工复查数据构建。
- `v0.3.1`：指基于 `v0.3` 数据训练出的最新版分类器模型：
  `chembl_classifier/models/chembl_classifier_selected_v0_3_1_scaffold_strongpoly.pt`

## 核心结论

- v0.3 新增生成分子去重后 `40218` 个，其中 `bad=30630`、`good=5029`、`ambiguous=4559`。
- 主要负样本来源是 fused heterocycle / large ring / saturated polycycle / three-member ring 等复杂环问题。
- v0.3.1 在测试集上指标很高：accuracy `0.972`、ROC-AUC `0.997`、negative accuracy `0.983`。
- 但对生成分子打分偏严格：v0.3 新生成分子中 good 的均值概率约 `0.633`，仍有约 `34.7%` good 被打到 `<0.5`；DiffSBDD/DiffGui 原始生成分子的中位数概率都约 `0.014`，说明这个分类器更像强过滤器，不适合直接作为唯一奖励。

## 图件清单

| 文件 | 看什么 | 原始路径 |
|---|---|---|
| `01_v03_curation_primary_reason_counts.png` | v0.3 新分子被标成 bad/good/ambiguous 的主原因分布 | `analysis/classifier/classifier_v0_3_scaffold_curation_20260610/primary_reason_counts.png` |
| `02_v03_curation_source_label_counts.png` | 不同生成来源里 good/bad/ambiguous 的数量 | `analysis/classifier/classifier_v0_3_scaffold_curation_20260610/source_label_counts.png` |
| `03_v031_negative_category_failure_rate.png` | v0.3.1 对各类负样本的漏判率/失败率 | `chembl_classifier/models/evaluation_selected_v0_3_1_scaffold_strongpoly/chembl_classifier_selected_v0_3_1_scaffold_strongpoly_negative_category_failure_rate.png` |
| `04_v031_generated_molecule_score_distribution.png` | v0.3.1 对 v0.3 新生成分子 good/bad/ambiguous 的打分分布 | `analysis/classifier/classifier_generated_score_distribution_20260613/generated_molecule_classifier_score_distribution.png` |
| `05_v031_diffgui_vs_diffsbdd_distribution.png` | v0.3.1 对 DiffGui 和 DiffSBDD 原始生成分子的打分对比 | `analysis/classifier/diffgui_diffsbdd_classifier_v031_20260615/diffgui_vs_diffsbdd_classifier_v031_distribution.png` |
| `06_v021_baseline_bad_reason_fp_rate.png` | 旧 v0.1/v0.2/v0.2.1 在 curated pool 上的对照 | `analysis/classifier/classifier_v0_2_1_weighted_20260608/curated_pool_v0_1_v0_2_v0_2_1/bad_reason_fp_rate_v0_1_v0_2_v0_2_1.png` |
| `07_v021_base_diffsbdd_probability_distribution.png` | 旧 v0.2.1 对 DiffSBDD 原始生成的打分分布 | `analysis/classifier/classifier_v0_2_1_weighted_20260608/base_diffsbdd_260407_distribution/01_classifier_probability_distribution.png` |

## 建议用途

- PPT 单页讲 classifier：优先放 `04`，再配 `01` 或 `03`。
- 讲“分类器分得太狠”：优先放 `04` 和 `05`。
- 讲“为什么要继续改 reward，而不是只靠 classifier”：用 `05`，因为 DiffSBDD/DiffGui 原始生成大量集中在低分区。
