# DiffGui 强化学习统计清单

> ⚠️ **定位说明**：DiffGui RL 是**横评基线与历史/对照线**，不是当前主线
> （主线为 DiffSBDD 保留线的 ESRPO，见 [`管线总览.md`](管线总览.md)）。本文的统计口径
> 仅适用于 DiffGui 线自身的记录。

> 适用范围：DiffGui 作为生成底座时的 DDPO/PPO 强化学习训练、离线采样、复筛与阶段总结。
> 本文只定义同一任务/同一靶点上的强化学习评估口径；跨靶点泛化不作为常规统计项。

## 1. 记录位置

长期复用的统计规范放在本文档：

- `docs/DiffGui强化学习统计清单.md`

每次训练或分析的实际证据分三层保存：

| 层级 | 位置 | 内容 |
|---|---|---|
| 原始运行证据 | `runs/<run_name>/` | 日志、config snapshot、checkpoint、每轮 SDF/CSV、GPU/耗时记录 |
| 汇总分析证据 | `analysis/diffgui_rl/<topic>_<date>/` | 聚合 CSV/JSON、绘图脚本、PNG/SVG、必要时的独立 `report.md` |
| 叙事记录 | `work_records/<YYMM>/<YYMMDD>/` | 本次为什么跑、参数组合、主要结果、和预期差异、剩余风险 |

如果图被引用到 `work_records/` 的 Markdown 中，图片应复制到同一目录，用本地文件名引用，便于归档和转 PDF。

## 2. 每个正式训练必须保存的原始文件

每个 `runs/<run_name>/` 至少应能追溯以下文件或等价文件：

- `train.log` 或 rank 日志：训练过程、报错、epoch 进度。
- `config.yaml` / `resolved_config.yaml`：实际使用参数。
- `checkpoints/latest.pt` 与关键 epoch checkpoint。
- `training_summary.csv` 或等价训练摘要。
- `metrics.csv` 或等价标量曲线。
- 每轮或关键轮的 `generated.sdf`。
- 每轮或关键轮的 `molecules_with_rewards.csv`。
- 对接相关的 `docking_rewards.csv`、`docking_scored.sdf`。
- `training_filter_decisions.csv`：过滤、guard、丢弃原因。
- `gpu_memory_samples.csv` 或等价资源记录。

这些文件用于保证后续分析不是只看一张图，而是能回到每个分子的原始证据。

## 3. 常规统计项

### 3.1 训练过程

每个 epoch 或固定间隔统计：

| 类别 | 指标 |
|---|---|
| RL 优化 | total reward、各 reward 分量均值/中位数/top 分位数 |
| PPO 稳定性 | policy loss、value loss、entropy、KL/approx KL、clip fraction、gradient norm |
| 采样质量 | 生成数、RDKit valid 数、对接成功数、过滤后保留数 |
| 失败来源 | sanitize 失败、碎片、价态异常、bad bond、docking 失败、guard 拦截 |
| 资源 | epoch 耗时、GPU 显存峰值、有效吞吐 |

核心判断不是 reward 单调上升，而是 reward 上升时有效率、结构质量和分子属性是否同步守住。

### 3.2 对接与亲和力

在同一靶点、同一对接口径下统计：

| 指标 | 说明 |
|---|---|
| Vina Score | 直接评分，报告均值、中位数、top-10、top-100 |
| Vina Min | 局部优化后评分，如当前流程支持 |
| Vina Dock | 重对接后评分，如当前流程支持 |
| High Affinity rate | 以参考配体或固定阈值为界，报告比例和阈值来源 |
| ligand efficiency | 避免模型只靠增大分子刷 docking |
| docking success rate | 单独报告，不能只在成功对接分子上算均值 |

如果只做强化学习阶段评估，不需要额外扩展到其他靶点；重点是同一靶点内不同 reward 设置、checkpoint、epoch 和 top-N 的可比性。

### 3.3 分子基本质量

| 指标 | 说明 |
|---|---|
| RDKit validity | 基础有效性 |
| PoseBusters validity | 有条件时作为 3D 合理性补充 |
| completeness | 是否生成完整分子 |
| uniqueness | 去重后比例 |
| diversity | 指纹多样性，避免 top 分子塌缩 |
| novelty / similarity | 和训练/参考配体的相似度，按任务需要开启 |
| size | 原子数/重原子数，必须和 docking 一起看 |

### 3.4 成药性与理化属性

| 指标 | 说明 |
|---|---|
| QED | 类药性 |
| SA | 合成可及性 |
| logP | 疏水性漂移监控 |
| TPSA | 极性漂移监控 |
| HBA/HBD | 氢键能力 |
| Lipinski count | 五规则符合数 |
| formal charge | 电荷异常检查 |

这些指标需要同时报告均值、中位数和 top docking 分子子集，避免平均值正常但 top 分子已经漂移。

### 3.5 DiffGui RL 特异性病理监控

这是当前 DiffGui RL 最重要的护栏统计。

| 类别 | 指标 |
|---|---|
| 元素组成 | carbon fraction、hetero atom count、element diversity、卤素比例 |
| 环结构 | ring count、3-member rings、5/6-member rings、7+ rings、fused rings、complex rings |
| 芳香结构 | aromatic rings、aromatic atom fraction、合理芳香骨架保留率 |
| 结构异常 | fragments、bad bond length、steric clash、异常价态 |
| guard 行为 | chemistry guard 触发率、ring pathology 触发率、filter pass rate |

判断标准应写成“坏环/复杂环下降，合理芳香骨架不被压没，validity 和 docking 不恶化”，而不是只追求平均环数下降。

## 4. Top-N 与复筛统计

每个候选模型至少输出：

| 子集 | 用途 |
|---|---|
| all generated | 看整体分布是否变好 |
| valid only | 看有效分子的真实分布 |
| docking success only | 看对接成功分子的亲和力 |
| top-100 by Vina | 对标 best-of-N，但必须同步报告 size/diversity |
| top-10 by Vina | 展示最强候选，但不能替代整体统计 |
| top quality-balanced | 综合 docking、QED、SA、guard、diversity 后的候选池 |

Top-N 表必须同时报告 Vina、QED、SA、diversity、size、guard failure rate。否则很容易把“更大、更油、更复杂”的分子误判为强化学习成功。

## 5. 消融与稳定性统计

强化学习阶段不做泛化，但需要做同一任务内的对照。

| 对照 | 目的 |
|---|---|
| 原始 DiffGui checkpoint | 判断 RL 是否真的超过底座 |
| reward-only 采样或离线评分 | 区分 reward 设计问题和 PPO 训练问题 |
| docking-only | 判断 docking 奖励本身带来的漂移 |
| docking + QED/SA | 判断属性奖励是否能守住成药性 |
| docking + ring pathology | 判断坏环是否下降 |
| docking + aromatic scaffold | 判断是否保留合理芳香骨架 |
| chemistry guard on/off | 判断护栏是否只是提高过滤率，还是改善生成分布 |
| freeze 策略 | 比较 last1/last2/last3 或其他冻结方案 |
| epoch/checkpoint | 比较 early/mid/final，识别后期漂移 |
| seed | 最终候选配置做 3 个以上 seed，报告 mean ± std |

每个对照都要写清楚“参数组合 -> 预期作用 -> 实际结果 -> 和预期差异 -> 是否继续”。

## 6. 重评估与论文级统计

以下项目不要求每个小实验都跑，只对最终 1-2 个候选模型或关键 checkpoint 做：

| 指标 | 使用场景 |
|---|---|
| redocking RMSD <= 2 A | 检查 pose 是否稳定 |
| strain energy | 检查构象是否过度紧张 |
| steric clashes | 检查空间冲突 |
| 键长/键角/二面角 JS 散度 | 对标 DiffGui 论文式几何质量 |
| 环大小分布 | 对标 DiffGui 并检查七元环/复杂环倾向 |
| MMGBSA | 只用于少量代表性 top 分子 |
| 3D 相互作用/药效团重叠 | 用于最终展示图和机制解释 |

这些属于重成本后评估，不应阻塞日常训练筛选。

## 7. 阶段总结建议表格

阶段总结报告至少包含三张表和三类图。

### 7.1 主结果表

| alias | reward 设置 | epoch | N | valid | dock success | Vina Score | Vina Dock | QED | SA | Div | Size | bad ring rate | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

### 7.2 Top-N 表

| alias | subset | N | Vina Score | Vina Dock | high affinity | QED | SA | Div | Size | guard pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 7.3 消融解释表

| alias | 改了什么 | 预期 | 实际结果 | 主要问题 | 下一步 |
|---|---|---|---|---|---|

### 7.4 图

- 训练曲线：reward 分量、validity、docking success、PPO 稳定性。
- 属性曲线：QED、SA、logP、TPSA、size、ring pathology。
- 最终热图：各 run 的核心指标，用短 alias 而不是长目录名。

## 8. 最小完成标准

一个 DiffGui RL 正式实验如果要进入阶段总结，至少需要满足：

1. 能追溯 config、log、checkpoint 和原始分子文件。
2. 有 all/valid/top-N 三个层面的 docking、QED、SA、diversity、size。
3. 有坏环、复杂环、芳香骨架、碳比例、杂原子、fragment 的护栏统计。
4. 有和原始 DiffGui checkpoint 或最近 baseline 的同靶点对比。
5. 有“参数组合 -> 结果 -> 和预期差异”的简表。
6. 图表使用短 alias，并在 `work_records/` 中复制同目录图片后引用。

