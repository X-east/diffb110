# DiffSBDD + 离线 IQL 中期答辩整体梳理：模型横评、数据基础与正式训练路线

> 最后更新：2026-09-29  
> 本记录是独立的 `DiffSBDD + offline IQL` 版本。**生成横评和独立生成评价仍包含 DiffGui**；只有强化学习训练（轨迹、actor、critic、BC/AWR/IQL）不使用 DiffGui。历史 DiffGui PPO/DDPO 不作为本路线的训练方法或训练证据。

## 1. 一句话主线

本研究先在 RE-CmeB/8GJL-C1101 上用固定尝试数比较 DiffGui、DiffSBDD、TargetDiff、PMDM 和 GCDM-SBDD 的生成、几何、靶点打分和化学空间表现，再把 DiffSBDD 冻结行为策略的完整去噪轨迹构造成离线 MDP，通过 `Original → BC-all → filtered BC → AWR → trajectory IQL` 的同预算对照，判断 IQL 是否能在**实际好分子率、结构合法性、靶点结合倾向和化学空间稳定性**之间带来可复现改善。

横评是模型起点证据，不是 IQL 结果；100k 轨迹是 IQL 的数据基础，不等于 actor 已经训练成功。当前还没有证据可以声称 IQL 已优于原始 DiffSBDD。

## 2. 当前路线与范围

### 2.1 正式技术路线

```text
五模型固定尝试数生成横评
  -> 选定 DiffSBDD 作为离线数据承载模型
  -> 100,000 条完整反向扩散轨迹
  -> finite reward sidecar + episode index + scaffold/no-leakage split
  -> Original / BC-all / filtered BC / AWR / trajectory IQL
  -> held-out critic go/no-go
  -> 新 checkpoint 固定尝试数生成
  -> RDKit / PoseBusters / 完整重对接 / 结构病理 / 多样性 / actual-good
  -> 只有联合指标通过才保留 IQL
```

### 2.2 训练与生成的边界

- **强化学习训练不使用 DiffGui**：IQL 数据来自 DiffSBDD 完整去噪轨迹；actor、critic、BC-all、filtered BC、AWR 和 IQL 都只围绕 DiffSBDD 实现。
- **生成横评和独立生成评价保留 DiffGui**：DiffGui 作为预训练生成基线参与五模型 fixed-attempt generation、PB-valid、QVina、完整重对接和化学空间比较。
- 不把 DiffGui PPO/DDPO 的历史训练轨迹或 checkpoint 混入 DiffSBDD IQL；也不把在线 DiffSBDD DDPO 的历史训练曲线当作当前 IQL 结果。
- 不把记录中的行为分数当成离线 PPO 的 old-policy log-prob；当前离线算法是 IQL、BC 和 AWR。
- 不把训练数据中的 top 分子重新统计为“新模型生成提升”。所有 DiffSBDD 控制或 IQL checkpoint 必须重新固定尝试数生成。

## 3. 模型横评：正式实验口径

正式横评使用 8GJL/C1101、1000 个扩散步、每模型 3 个独立重复、每重复 1000 次固定尝试，共 3000 次尝试/模型；失败不补采样。五个生成模型为：`DiffGui`、`DiffSBDD`、`TargetDiff`、`PMDM`、`GCDM-SBDD`。其中 DiffGui 只承担生成基线和独立评价角色，不进入 IQL 训练。

评价层级必须分开解释：

1. **SDF 写出**：是否写出记录，不等于分子有效；
2. **sanitize-valid / unique SMILES**：化学可读性与重复情况；
3. **PB-valid**：几何和结构合法性；
4. **QVina score-only**：保留原始生成姿势的固定姿势打分；
5. **UniDock 完整重对接**：统一片段预处理后重新搜索姿势；
6. **actual-good**：未来正式 IQL 评价中的联合门槛，不由本次预训练横评单项指标替代。

### 3.1 生成与合法性漏斗

下表为三重复合并成功数；百分比是每批 1000 次尝试的成功率均值 ± 样本标准差（`n=3`）。

| 模型 | SDF 写出 | sanitize-valid | unique SMILES | PB-valid | QVina score-only 成功 | 完整重对接成功 |
|---|---:|---:|---:|---:|---:|---:|
| **DiffGui** | 2641/3000；88.03% ± 1.25% | 2641/3000；88.03% ± 1.25% | 2610/3000；87.00% ± 1.35% | **2522/3000；84.07% ± 0.68%** | 2641/3000；88.03% ± 1.25% | 2641/3000；88.03% ± 1.25% |
| **DiffSBDD** | 3000/3000；100.00% ± 0.00% | 1999/3000；66.63% ± 0.65% | 1999/3000；66.63% ± 0.65% | 1309/3000；43.63% ± 1.00% | 1805/3000；60.17% ± 0.74% | 1999/3000；66.63% ± 0.65% |
| **TargetDiff** | 2974/3000；99.13% ± 0.06% | 2974/3000；99.13% ± 0.06% | 2972/3000；99.07% ± 0.06% | 2182/3000；72.73% ± 0.40% | 2931/3000；97.70% ± 0.17% | 2974/3000；99.13% ± 0.06% |
| **PMDM** | 1726/3000；57.53% ± 0.91% | 1726/3000；57.53% ± 0.91% | 1726/3000；57.53% ± 0.91% | 1359/3000；45.30% ± 1.11% | 1726/3000；57.53% ± 0.91% | 1726/3000；57.53% ± 0.91% |
| **GCDM-SBDD** | 3000/3000；100.00% ± 0.00% | 1570/3000；52.33% ± 1.21% | 1557/3000；51.90% ± 0.98% | 895/3000；29.83% ± 0.90% | 1570/3000；52.33% ± 1.21% | 1570/3000；52.33% ± 1.21% |

![五模型固定尝试数生成与合法性漏斗](../../work_records/images/five_model_generation_funnel.png)

**横评解释：** TargetDiff 的生成成功率和 DiffGui 的 PB-valid 尝试级率分别突出；DiffSBDD 的 SDF 写出率为 100%，但 sanitize-valid 只有 66.63%，因此不能用“SDF 条目写满”代替有效分子率。所有模型的后续评价必须始终保留 fixed-attempt denominator。

### 3.2 原始姿势 QVina score-only

score-only 不重新搜索位置和构象，回答的是“模型直接生成的三维姿势是否已经处在有利口袋区域”。多片段/打分准备失败不应被隐藏在成功分母中。

| 模型 | 成功/尝试 | 分数均值 | 中位数 | 最佳 5% 均值 | `<= -10 kcal/mol` |
|---|---:|---:|---:|---:|---:|
| **DiffGui** | 2641/3000；88.03% ± 1.25% | **-9.605 ± 0.036** | **-9.774 ± 0.019** | -13.120 ± 0.151 | **45.05% ± 0.72%** |
| **DiffSBDD** | 1805/3000；60.17% ± 0.74% | -4.237 ± 0.434 | -5.837 ± 0.052 | -9.637 ± 0.088 | 1.16% ± 0.28% |
| **TargetDiff** | 2931/3000；97.70% ± 0.17% | -9.090 ± 0.037 | -9.051 ± 0.038 | **-13.626 ± 0.164** | **33.74% ± 0.41%** |
| **PMDM** | 1726/3000；57.53% ± 0.91% | 4.353 ± 2.869 | -2.454 ± 0.289 | -9.125 ± 0.437 | 1.05% ± 0.48% |
| **GCDM-SBDD** | 1570/3000；52.33% ± 1.21% | 25.911 ± 1.334 | 16.480 ± 0.921 | -2.519 ± 0.178 | 0.00% ± 0.00% |

### 3.3 UniDock 完整重对接

完整重对接允许统一片段预处理后重新搜索姿势，回答的是“生成分子能否重新找到有利结合姿势”，不能反向证明原始三维姿势质量。

| 模型 | 成功/尝试 | 中位数 | 最佳 5% 均值 | `<= -10 kcal/mol` |
|---|---:|---:|---:|---:|
| **DiffGui** | 2641/3000；88.03% | -11.009 ± 0.024 | -14.283 ± 0.183 | **71.30% ± 0.48%** |
| **DiffSBDD** | 1999/3000；66.63% | -9.580 ± 0.022 | -12.829 ± 0.099 | 39.07% ± 0.68% |
| **TargetDiff** | 2974/3000；99.13% | **-11.087 ± 0.084** | **-15.214 ± 0.106** | **70.11% ± 1.95%** |
| **PMDM** | 1726/3000；57.53% | -9.663 ± 0.135 | -12.795 ± 0.248 | 41.06% ± 3.93% |
| **GCDM-SBDD** | 1570/3000；52.33% | -6.526 ± 0.059 | -10.269 ± 0.201 | 2.48% ± 0.29% |

![五模型靶点评价：原始姿势与完整重对接](../../work_records/images/five_model_docking_comparison.png)

**横评结论：** TargetDiff 在生成覆盖和完整重对接上领先，DiffGui 在 PB-valid、原始姿势 score-only 和完整重对接阈值比例上突出；DiffSBDD 不是单项 docking 第一，但其化学空间参考重叠较强，且已经拥有完整冻结去噪轨迹，适合作为当前离线 IQL 的数据承载模型。这个选择是“数据可训练性 + 同靶点评价闭环”的路线选择，不是宣称 DiffSBDD 预训练模型全面领先。

### 3.4 完整重对接 Top 10 结构展示

下列结构只用于展示横评中各模型的高 docking 分子，不是 IQL 最终候选。Top 10 仅经过 sanitize-valid、最大片段处理和完整重对接成功筛选，尚未经过正式 actual-good 联合门槛。

| 模型 | Top 10 分数范围 | 中位数 | MW 中位数 | LogP 中位数 | QED 中位数 | TPSA 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| **DiffGui** | -16.718 至 -15.294 | -15.534 | 441.57 | 5.86 | 0.514 | 45.82 |
| **DiffSBDD** | -14.549 至 -13.701 | -13.982 | 457.33 | 5.83 | 0.526 | 59.56 |
| **TargetDiff** | -17.827 至 -16.422 | **-16.738** | 484.54 | **6.37** | 0.395 | 27.61 |
| **PMDM** | -15.037 至 -13.456 | -13.802 | 316.40 | 3.79 | **0.707** | 36.53 |
| **GCDM-SBDD** | -12.789 至 -11.193 | -11.435 | 382.48 | 2.19 | 0.457 | 79.68 |

#### DiffSBDD

![DiffSBDD 完整重对接 Top 10](../../work_records/images/top10_2d_diffsbdd.png)

#### TargetDiff

![TargetDiff 完整重对接 Top 10](../../work_records/images/top10_2d_targetdiff.png)

#### PMDM

![PMDM 完整重对接 Top 10](../../work_records/images/top10_2d_pmdm.png)

#### GCDM-SBDD

![GCDM-SBDD 完整重对接 Top 10](../../work_records/images/top10_2d_gcdm_sbdd.png)

### 3.5 与 ChEMBL36 参考空间的化学空间降维（100k 主图 + 全体扩展）

本节明确区分两个参考层级，避免把 `100,000` 个参考分子和 ChEMBL36 的完整筛选集混为一谈。两套图均使用 ECFP4（`radius=2`、2048 bit）、cosine UMAP、`n_neighbors=20`、`min_dist=0.1`、`random_state=42`，五模型均为 3 次重复、每次 1,000 个固定尝试，不补齐失败。

此外，在两张分面图的 **ChEMBL36 参考子图** 中叠加了项目外部整理的 RND-family 外排泵抑制剂（EPI）参考集（`n=53`，黑色白边菱形）；五个模型子图不叠加这些点。该集 53/53 可 RDKit sanitize，canonical SMILES 均唯一；其与 ChEMBL36 100k 子集和 1,550,090 全体筛选集的精确结构重叠分别为 3 和 17 个，与五模型 sanitize-valid 生成分子的精确结构重叠为 0。黑色菱形是外部 RND 参考化合物，不是本次生成分子，也不能视为 CmeB 特异阳性样本；其作用仅是为生成分子相对已知 RND EPI 的结构空间位置提供可视化参照。

#### 3.5.1 参考集数量口径

| 层级 | 分子数 | 含义 |
|---|---:|---|
| ChEMBL36 原始输入记录 | 2,854,815 | ChEMBL36 清洗表在项目过滤前的记录数；不是本次 UMAP 直接使用的参考集 |
| ChEMBL36 项目筛选后的完整参考集 | **1,550,090** | 去重后、分子量 150--500、重原子数 10--30、非带电的全部保留分子；这是“全体”图的参考集 |
| ChEMBL36 固定 100k 子集 | 100,000 | 从上述 1,550,090 条中按固定 seed=42 抽取，用于之前的主图及全体扩展图的 UMAP 拟合 landmarks |
| 五模型 sanitize-valid 生成分子 | 10,910 | DiffGui/DiffSBDD/TargetDiff/PMDM/GCDM-SBDD 为 2,641/1,999/2,974/1,726/1,570 |

因此，`1,550,090` 是**项目过滤口径下的 ChEMBL36 全体**，不是 ChEMBL36 原始数据库的全部条目；原始输入记录数为 `2,854,815`。`100,000` 是从 155 万筛选集抽出的参考子集。

> ⚠️ **口径更新（2026-09-28 起）**：ChEMBL 参照集已重建为 **1,790,085**（Small molecule + 最大片段 +
> 中性 + 10 元素白名单 + 去重）；R1–R5 全清率由 84.72% 修正为 **80.94%**（HA10–30 口径 = 90.60%）。
> 本节的 `1,550,090` / 155 万 属 2026-08 历史口径，仅用于理解当时的 IQL 数据构建，
> 引用时请以新参照集为准（[260928-03](../work_records/26-9/260928/260928-03 ChEMBL新参照集R1-R5合理率重算.md)）。

#### 3.5.2 原先的 100k 主图（保持之前图样式）

这组图是在 `100,000` 条固定 ChEMBL36 子集与五模型 `10,910` 条有效生成分子上联合拟合一套二维坐标，作为模型间主对照图。图中 ChEMBL 背景明确标注为 `n=100,000`。

![五模型与 ChEMBL36 100k 联合化学空间总览](../../work_records/images/five_model_chembl100k_umap_overview.png)

![五模型与 ChEMBL36 100k 联合化学空间分面](../../work_records/images/five_model_chembl100k_umap_facets.png)

![五模型到 ChEMBL36 100k 的 UMAP 距离汇总](../../work_records/images/five_model_chembl100k_umap_distance_summary.png)

![100k 分面图仅在 ChEMBL36 子图叠加 53 个 RND 家族 EPI 参考分子](../../work_records/images/five_model_chembl100k_umap_facets_rnd53.png)

| 模型 | 质心到 ChEMBL 距离 | 最近 ChEMBL 距离中位数 | ChEMBL 自身近邻 P95 范围覆盖率 |
|---|---:|---:|---:|
| DiffGui | 2.911 ± 0.017 | 0.0179 ± 0.0003 | 63.10% ± 1.80% |
| **DiffSBDD** | **2.299 ± 0.032** | **0.0139 ± 0.0006** | **73.34% ± 0.32%** |
| PMDM | 3.967 ± 0.074 | 0.0366 ± 0.0012 | 36.03% ± 1.37% |
| GCDM-SBDD | 4.290 ± 0.045 | 0.0252 ± 0.0019 | 48.79% ± 4.11% |
| TargetDiff | 3.031 ± 0.049 | 0.0210 ± 0.0002 | 56.93% ± 0.57% |

100k 图的逐分子和汇总数据：

| 数据 | 内容 | 链接 |
|---|---|---|
| `joint_umap_coordinates.csv.gz` | 100,000 个 ChEMBL + 10,910 个生成分子的坐标和身份字段 | [joint_umap_coordinates.csv.gz](../../../analysis/benchmark/benchmark5/five_model_chembl100k_umap_20260730/joint_umap_coordinates.csv.gz) |
| `group_summary.csv` | 六组样本数、唯一数、质心和组内 RMS 半径 | [group_summary.csv](../../../analysis/benchmark/benchmark5/five_model_chembl100k_umap_20260730/group_summary.csv) |
| `distance_to_chembl_summary.csv` | 五模型三重复距离和覆盖率汇总 | [distance_to_chembl_summary.csv](../../../analysis/benchmark/benchmark5/five_model_chembl100k_umap_20260730/distance_to_chembl_summary.csv) |
| `nearest_chembl_distance_per_generated_molecule.csv.gz` | 10,910 条生成分子的最近 ChEMBL 距离 | [nearest_chembl_distance_per_generated_molecule.csv.gz](../../../analysis/benchmark/benchmark5/five_model_chembl100k_umap_20260730/nearest_chembl_distance_per_generated_molecule.csv.gz) |
| `manifest.json` | 100k 来源、固定 seed、参数、数量和解释边界 | [manifest.json](../../../analysis/benchmark/benchmark5/five_model_chembl100k_umap_20260730/manifest.json) |

RND-family EPI 参考点的坐标和重叠审计：

| 数据 | 内容 | 链接 |
|---|---|---|
| `rnd53_coordinates_chembl100k.csv` | 53 个参考分子在历史 100k UMAP 坐标系中的坐标与身份字段 | [rnd53_coordinates_chembl100k.csv](../../../analysis/benchmark/benchmark5/rnd53_umap_overlay_20260817/rnd53_coordinates_chembl100k.csv) |
| `rnd53_coordinates_chemblfull.csv` | 53 个参考分子在全体筛选集扩展图坐标系中的兼容投影坐标 | [rnd53_coordinates_chemblfull.csv](../../../analysis/benchmark/benchmark5/rnd53_umap_overlay_20260817/rnd53_coordinates_chemblfull.csv) |
| `rnd53_overlap_audit.json` | 53 个参考分子的 RDKit 有效性、唯一性及与 ChEMBL/生成集的精确重叠审计 | [rnd53_overlap_audit.json](../../../analysis/benchmark/benchmark5/rnd53_umap_overlay_20260817/rnd53_overlap_audit.json) |
| `old100k_coordinate_reproduction_audit.json` | 100k 历史图的输入与坐标复现审计 | [old100k_coordinate_reproduction_audit.json](../../../analysis/benchmark/benchmark5/rnd53_umap_overlay_20260817/old100k_coordinate_reproduction_audit.json) |

#### 3.5.3 ChEMBL36 全体筛选集扩展图

全体扩展图保留 100k landmarks 与五模型共同拟合的坐标系，然后把其余 ChEMBL36 筛选集分子 transform 到同一坐标系；不会重新拟合一套与 100k 主图不可比的新 UMAP。这样可以在不改变五模型坐标的情况下显示完整的 `1,550,090` 条筛选参考分布。

![五模型与全体 ChEMBL36 筛选集的联合 UMAP 总览](../../work_records/images/five_model_chemblfull_umap_overview_v001.png)

![五模型与全体 ChEMBL36 筛选集的 UMAP 分面图](../../work_records/images/five_model_chemblfull_umap_facets_v001.png)

![五模型全体 ChEMBL36 覆盖率与最近距离汇总](../../work_records/images/five_model_chemblfull_umap_summary_v001.png)

![全体 ChEMBL36 分面图仅在 ChEMBL36 子图叠加 53 个 RND 家族 EPI 参考分子](../../work_records/images/five_model_chemblfull_umap_facets_rnd53.png)

| 模型 | ChEMBL36 95% HDR 覆盖率 | 最近 ChEMBL UMAP 距离中位数 |
|---|---:|---:|
| **DiffGui** | 61.41% ± 1.37% | 0.004272 ± 0.000196 |
| **DiffSBDD** | **64.18% ± 1.32%** | **0.003823 ± 0.000093** |
| **TargetDiff** | 44.28% ± 1.03% | 0.005991 ± 0.000185 |
| **PMDM** | 20.68% ± 2.28% | 0.009993 ± 0.001006 |
| **GCDM-SBDD** | 20.58% ± 1.99% | 0.007067 ± 0.000130 |

全体扩展图的坐标、统计和复现文件：

| 数据 | 内容 | 链接 |
|---|---|---|
| `manifest.json` | 全体数量、100k landmarks、输入哈希、UMAP 参数和解释边界 | [manifest.json](../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/manifest.json) |
| `population_counts.csv` | ChEMBL 与五模型纳入数、固定尝试数和唯一率 | [population_counts.csv](../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/population_counts.csv) |
| `coverage_summary.csv` | 五模型三重复 HDR 覆盖率汇总 | [coverage_summary.csv](../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/coverage_summary.csv) |
| `distance_to_chembl_summary.csv` | 五模型三重复最近距离汇总 | [distance_to_chembl_summary.csv](../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/distance_to_chembl_summary.csv) |
| `generated_coordinates.csv.gz` | 10,910 条生成分子的 UMAP1/UMAP2 与来源身份 | [generated_coordinates.csv.gz](../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/generated_coordinates.csv.gz) |
| `generated_embedding_metrics.csv.gz` | 10,910 条生成分子的坐标、HDR 标记和最近 ChEMBL 距离 | [generated_embedding_metrics.csv.gz](../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/generated_embedding_metrics.csv.gz) |
| `chembl_coordinates.csv.gz` | 全体筛选集 1,550,090 条 ChEMBL36 分子的固定坐标 | [chembl_coordinates.csv.gz](../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/chembl_coordinates.csv.gz) |
| `chembl_self_distance_reference.csv` | ChEMBL 自身近邻距离参考 | [chembl_self_distance_reference.csv](../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/chembl_self_distance_reference.csv) |
| `umap_model.joblib` | 固定 UMAP 模型，可用于同口径 transform | [umap_model.joblib](../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/umap_model.joblib) |

两组指标必须分别解读：100k 主图用于延续之前的模型对照，1,550,090 全体筛选集图用于展示完整过滤参考分布。二者都只是固定 ECFP4-UMAP 投影中的邻域重叠，不能直接解释为靶点药效、对接质量或绝对分子质量；对 IQL 更适合作为分布漂移监控和结构病理预警，而不是单独的正样本标签。

对 RND 叠加层，100k 图使用与历史 100k 图相同的 100,000 个 ChEMBL 和 10,910 个生成分子重新拟合并复现坐标，历史坐标的最大绝对差为 `8.83e-7`（RMSE `1.87e-7`），因此 ChEMBL36 子图可与原主图直接对照。全体图所存 UMAP 模型因远端 Python/UMAP/numba 序列化版本差异无法在当前环境直接载入，故 ChEMBL36 子图中的 53 个参考点采用原始稀疏 ECFP4 cosine 近邻的兼容投影；20 个已保存非 landmark ChEMBL 点的验证中位欧氏误差为 `0.105`，但最大误差为 `5.873`。因此这些 RND 菱形仅限于 ChEMBL36 参考子图中的整体视觉定位，不能用于逐分子距离排序、最近邻定量或任何活性推断。

### 3.6 结构病理风险

| 模型 | 低杂原子富碳 | Brenk 命中 | SA > 6 | 环数 >= 6 |
|---|---:|---:|---:|---:|
| **DiffGui** | 8.25% | 25.95% | 6.73% | 23.06% |
| **DiffSBDD** | 3.00% | 58.44% | 6.60% | 5.95% |
| **TargetDiff** | 15.70% | 40.35% | 9.35% | 13.48% |
| **PMDM** | 6.48% | 44.49% | **23.94%** | **23.69%** |
| **GCDM-SBDD** | 4.84% | **75.86%** | 1.90% | 0.00% |

![五模型结构病理风险指标](../../work_records/images/five_model_structure_risk.png)

横评提示五个模型具有不同的风险方向：DiffSBDD 的 Brenk 命中偏高，PMDM 的 SA 和复杂环风险更突出，GCDM-SBDD 的 Brenk 命中最高，DiffGui 的复杂环风险更明显，TargetDiff 的富碳/多环风险更突出。因此 IQL reward 不能只优化 docking 或单一 classifier；结构合理性必须以独立标签和连续监控共同约束。

## 4. DiffSBDD 100k 完整轨迹：IQL 的数据基础

### 4.1 已完成的冻结行为策略采集

正式目录：

`/data/disk6/DiffB110/runs/diffsbdd_collect/diffsbdd-original-t1000-8gpu-100k-trajectory_20260730_194643`

| 项目 | 已验证结果 |
|---|---:|
| 固定尝试数 | 100,000 |
| 完整 shard | 10,000 / 10,000 |
| GPU/rank | 8 |
| 每 rank shard | 1,250 |
| 每 episode 扩散步 | 1,000 |
| 每 episode 动作数 | 1,001 |
| OpenBabel 重建 | 100,000 |
| RDKit-valid episode | 67,694 |
| 无效 episode | 32,306 |
| 轨迹字节数 | 157,847,462,576 |
| seeds | 1--8（按 GPU/rank） |
| 完成状态 | `complete`，errors 为空 |

![100k 轨迹初始分片指标](../../work_records/images/initial_shard_metrics.png)

完成证据包括 `summary.json`、`completion_report.json` 和 `run_manifest.json`。这证明的是固定 8GJL/C1101 条件下的轨迹采集完成，不代表跨靶点泛化，也不代表 reward sidecar 或 IQL actor 已完成。

### 4.2 Episode index 与数据分布

完整 episode index 已建立，包含 100,000 个 episode；其中 valid/invalid 为 67,694/32,306，节点数范围为 10--30。当前 index 快照明确标记为 `formal_split=false`，因为尚未使用正式 reward strata；因此不能把它直接称为最终模型选择 split。

![100k episode index 检查](../../work_records/images/index_smoke_summary.png)

## 5. 离线 IQL 方法设计

### 5.1 有限时域 MDP

每个逆扩散决策点使用：

```text
state  = (z_t, t, pocket, node_count, masks)
action = z_{t-1}
next   = action-derived next state
reward = ordinary finite reward + terminal episode reward
```

最终 decode 不是普通 reverse step：坐标来自 `xh_final.xyz`，原子类型来自 `argmax(z0.h)`，最终 actor action mask 只覆盖 xyz。因此实现中必须保留 ordinary step 与 final decode 的边界。

### 5.2 为什么不用离线 PPO

已有记录的行为分数存在裁剪和饱和，不能安全还原为 PPO old-policy log-prob。将它们直接放进 `ratio = exp(logp_new - logp_old)` 会把数值分数误当成概率证据。当前采用：

- `BC-all`：所有可用行为轨迹等权模仿；
- `filtered BC`：只模仿严格 positive/actual-good 轨迹；
- `AWR`：基于 episode return 的加权模仿；
- `IQL`：expectile V、Twin-Q，以及 `exp(beta * (Q - V))` actor 权重。

### 5.3 训练对照矩阵

| 方法 | Q/V critic | actor 训练 | 作用 |
|---|---|---|---|
| Original | 无 | 不训练 | 冻结 DiffSBDD 基线 |
| BC-all | 无 | 全部行为数据，权重 1 | 判断普通行为模仿效果 |
| filtered BC | 无 | 仅 strict positive | 判断正样本稀缺时的模仿上限 |
| AWR | 无 | 标准化 episode return | 低复杂度加权基线 |
| trajectory IQL | Twin-Q + V | `exp(beta * (Q - V))` | 当前主方法，抑制 OOD action 外推 |

所有方法必须固定：数据版本、episode split、actor trainable layers、学习率、训练步数、batch、checkpoint cadence、采样 attempts 和 independent evaluation funnel。

## 6. 已完成工程证据与当前限制

### 6.1 已有的工程证据

- 完整 100k 轨迹、10k shards、manifest 和固定尝试分母已经完成。
- episode index、shard-local time-stratified sampler、reward sidecar 关联、几何 Twin-Q/V 和 BC/AWR/IQL 更新闭环已有实现记录。
- 单步真实 DiffSBDD actor smoke 可产生 finite loss 和参数变化。
- runner CUDA smoke 完成 2 个 epoch、2 个 checkpoint，并验证 checkpoint restore/resume；但该 smoke 使用 validity interface probe，不是正式分子质量实验。
- 离线 RL 测试记录为 `139 passed, 3 skipped`；这是此前独立 worktree 的测试证据，本次未重新运行整套测试。

![IQL 单步训练 loss smoke](../../work_records/images/iql_train_step_losses.png)

![IQL runner checkpoint/resume smoke](../../work_records/images/iql_runner_resume_losses.png)

### 6.2 不能写成正式结果的内容

目前不能声称：

1. IQL actor 已完成正式训练；
2. IQL 已经提高 actual-good 率；
3. IQL 已经超过 Original、BC-all、filtered BC 或 AWR；
4. held-out critic 已经通过科学 go/no-go；
5. 已经有 IQL checkpoint 的独立固定尝试数生成结果。

## 7. 正式 reward、curation 与 go/no-go

### 7.1 reward 与 actual-good 分离

训练使用连续、finite 的 `reward_total_train`；最终科学判据使用独立的 `actual_good`：

```text
actual_good =
  sanitize-valid
  AND PB-valid
  AND complete docking success
  AND docking / ligand-efficiency threshold
  AND QED / SA / LogP / TPSA constraints
  AND no critical ring / pathology failure
  AND single main fragment
```

失败样本必须获得有限惩罚，不能写入 NaN/Inf。sidecar 还需保留重建、sanitize、RDKit、PB、docking、ligand efficiency、QED、SA、LogP、TPSA、元素/环结构、fragment、failure reason、reward version 和 provenance。

### 7.2 现有 curation smoke 的含义

80 episode curation smoke 的类别计数为：positive 1、property_negative 0、structural_negative 77、target_negative 2、technical_failure 0；split 为 64/8/8，唯一 positive scaffold 落在 train。该结果只证明 curation 和 split 接口能运行，不能支持正式 filtered BC 或 held-out actor 结论。

![curation smoke 类别计数](../../work_records/images/curation_class_counts.png)

### 7.3 进入正式 actor 前的门槛

| 门槛 | 必须满足的证据 |
|---|---|
| 数据完整 | 10,000/10,000 shards，manifest/SHA 通过，100,000 episode 全覆盖 |
| reward 完整 | sidecar 100% 关联，所有 `reward_total_train` finite |
| split 无泄漏 | 按 episode/scaffold 分组，train/validation/test 不交叉 |
| positive 支持 | 审计 actual-good 数量、正 scaffold 数量及其在 validation/test 中的支持 |
| critic | held-out 排序、校准、时间分层误差和 Q1/Q2 disagreement 优于 node-count/validity baseline |
| actor | ESS、advantage 权重、参数漂移和 descriptor drift 可控 |
| 独立复评 | 每个 checkpoint 新生成 fixed attempts，并过完整评价漏斗 |

工程建议的 actual-good 解释区间：`>=500` 才可进入主实验；`200--499` 只能做偏弱的 critic/连续 reward 探索；`<200` 不启动全量 IQL actor。

![DiffSBDD + IQL 当前证据门槛](../../work_records/images/diffsbdd_iql_evidence_gates.png)

## 8. Held-out critic：当前只能作为 smoke

当前 v3 critic evaluation smoke 只覆盖 validation 8 episodes，记录：Spearman `0.503`、Pearson `0.215`、MAE `0.942`、RMSE `1.103`，且 residual 与 node count 的 Pearson 为 `0.743`；`scientific_gate_decision` 明确为 `not_assigned_by_evaluator`，scope 为 `heldout_critic_evaluation_not_actor_quality_claim`。

因此它只能说明评估链可以输出指标，不能证明 critic 已学会靶点质量排序。正式评估必须扩大到冻结 validation/test，报告 invalid/valid 分层、node-count 混淆、time-bin 分层、Q1/Q2 disagreement 和 calibration。

![held-out critic smoke](../../work_records/images/heldout_critic_evaluation.png)

## 9. 下一步执行顺序

1. 修复并重跑 formal label worker 的 RDKit 异常处理；不得复用未形成 final summary 的部分标签目录。
2. 完成 100k reward sidecar、finalize、hash 和全量 coverage audit。
3. 生成正式 curation manifest，统计 actual-good、positive scaffold、negative 类别和固定分母。
4. 冻结 scaffold-grouped 80/10/10 episode split，并检查 positive scaffold 是否覆盖 validation/test。
5. 先用相同预算训练 Original evaluation、BC-all、filtered BC、AWR 对照。
6. 通过 held-out critic go/no-go 后再启动 trajectory IQL actor。
7. 每个 checkpoint 按相同 seed/attempt 规则独立生成，评价 RDKit、PB、完整 docking、actual-good、性质、结构病理和多样性。
8. 只有当 IQL 在联合指标上超过 Original 与简单对照，且没有明显 descriptor/pathology 漂移，才保留 IQL 为正式路线。

## 10. 证据路径

### 10.1 主要原始数据

- 轨迹完成证据：`../../../runs/diffsbdd_collect/diffsbdd-original-t1000-8gpu-100k-trajectory_20260730_194643/summary.json`
- 轨迹完成报告：`../../../runs/diffsbdd_collect/diffsbdd-original-t1000-8gpu-100k-trajectory_20260730_194643/completion_report.json`
- 完整 episode index：`../../../analysis/offline_iql_full_index_260803/episode_index.jsonl`
- 轨迹元数据汇总：`../../../collected_iql_benchmark_100k_260806/trajectory_100k_metadata/`

### 10.2 横评数据

- 五模型漏斗：`../../../analysis/midterm_defense_data_audit_20260730/generation_funnel_repeat_summary.csv`
- 关键横评指标：`../../../analysis/midterm_defense_data_audit_20260730/benchmark_key_metrics.csv`
- 正式生成汇总：`../../../analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/formal_3rep_model_summary.csv`
- 正式完整重对接汇总：`../../../analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/formal_3rep_docking_model_summary.csv`
- PoseBusters 汇总：`../../../analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/posebusters_queue/posebusters_model_summary.csv`
- 全体 ChEMBL36 降维结果：`../../../analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001/`

### 10.3 IQL 工程与 smoke

- 完整 index 诊断：`../../../analysis/offline_iql_full_index_260803/`
- curation smoke：`../../../analysis/offline_iql_curation_smoke_v1_260803/`
- held-out critic smoke：`../../../analysis/offline_iql_critic_eval_smoke_v3_260803/`
- runner resume smoke：`../../../analysis/offline_iql_runner_smoke_260731_r2/`
- 单步训练 smoke：`../../../analysis/offline_iql_train_step_smoke_260730/`
- 图表脚本：`../../../analysis/diffsbdd_iql_midterm_20260817/plot_diffsbdd_iql_midterm.py`

### 10.4 DiffSBDD 生成分子投影到旧五模型 UMAP

为保持与五模型历史横评图的坐标一致，使用原有的 `100,000` 条 ChEMBL 参考及五模型生成分子拟合的 UMAP reducer，对 DiffSBDD 100,000 次生成中的 RDKit-valid **终态分子**执行 `transform`。这里投影的对象是分子，不是整条去噪轨迹；`trajectory_id` 仅作为分子到原始 `.pt` 轨迹、SDF 和重建 CSV 的可追溯主键。

| 项目 | 数量 |
|---|---:|
| DiffSBDD 生成尝试 | 100,000 |
| RDKit-valid 终态分子 | 67,694 |
| RDKit-invalid | 32,306 |
| trajectory_id 重复 | 0 |
| 非有限 UMAP 坐标 | 0 |

![DiffSBDD 生成分子投影到旧五模型 UMAP](../../work_records/images/diffsbdd_molecules_projected_on_old_five_model_umap.png)

原始专项记录：`../260818/260818-01 DiffSBDD轨迹投影到旧五模型UMAP.md`（标题虽保留旧文件名，正文已明确投影对象为生成分子）。

投影坐标：`../../../analysis/offline_iql_chembl100k_umap_old_reference_retry_260817/diffsbdd_trajectory_projected_coordinates.csv.gz`。

该图仅用于化学空间定位，不能替代原始 ECFP4 最近邻相似度、结构警示、PoseBusters、对接和 actual-good 联合筛选。

## 11. 当前结论

当前最稳妥的答辩表述是：

> 在 RE-CmeB/8GJL-C1101 上，五模型固定尝试数横评显示 TargetDiff 在生成覆盖和完整重对接上较强，DiffGui 在 PB-valid 和原始姿势 score-only 上突出；DiffSBDD 已形成完整、可审计的 100k 反向扩散轨迹，并在 ECFP4-UMAP 化学空间参考中表现出较高的 ChEMBL 邻域覆盖。因此，本研究将 DiffSBDD 作为离线数据承载模型，采用 BC、AWR 与 trajectory IQL 的同预算对照，重点验证“实际好分子联合通过率”是否改善；截至 2026-08-17，IQL actor、正式 sidecar、held-out critic go/no-go 和独立生成复评尚未完成，不能提前宣称 IQL 有效。
