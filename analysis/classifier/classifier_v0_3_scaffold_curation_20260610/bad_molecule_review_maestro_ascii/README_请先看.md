# 坏分子中文分类查看包

这个目录用于人工查看 v0.3 scaffold 分类里被标记为坏分子的分子。

注意：为了兼容 Maestro/Schrödinger，所有 SDF 文件名、SDF molecule name 和新增 SDF property 都使用英文/ASCII。
中文原因只写在 README 和 CSV 对照表里。

## 怎么看

- `00_all_bad_primary_unique.sdf`：每个坏分子只出现一次，适合整体浏览。
- `primary_unique/`：每个坏分子只进入一个主原因 SDF，适合建立最终负样本集合。
- `all_triggered_reasons/`：同一个分子如果触发多个规则，会出现在多个 SDF，适合检查某条规则是否过严。
- `坏分子分类总表.csv`：中文原因 + 描述符 + 原始来源，适合筛选和排序。
- 每个 SDF 分子属性里都保留 `primary_reason`、`all_reasons`、`reason_count`；一个分子可以有多个坏原因。

## 建议优先查看顺序

1. 无芳香环的多饱和碳环：确认当前 DiffGui RL 里“多个碳环但没有苯环”的坏模式。
2. 小芳香环过度卤代 / 卤素数量过多：确认五六个卤素、过小卤代芳环是否都应强负。
3. 稠合杂环/复杂杂环：确认哪些是明显不合理，哪些只是复杂但可接受。
4. 多稠合或桥环体系：确认桥环/螺环/大稠环是否需要一刀切还是降权。
5. 高碳低杂原子：确认是否只是分布偏差，而不是单分子绝对错误。

主原因不重复合并 SDF 分子数：30630

## 按主原因计数

| 文件 | 中文原因 | 强度 | 数量 | 说明 |
|---|---:|---:|---:|---|
| `primary_unique/01_bad_saturated_polycycle_no_aromatic_strong_negative.sdf` | 无芳香环的多饱和碳环 | 强负样本 | 2462 | 多个饱和碳环堆叠但没有苯环/芳香环，容易成为当前 DiffGui RL 的错误偏好。 |
| `primary_unique/02_bad_overhalogenated_small_aromatic_strong_negative.sdf` | 小芳香环过度卤代 | 强负样本 | 5 | 单个小芳香环上连太多卤素，尤其多 Br/Cl/F，分子过小且不自然。 |
| `primary_unique/03_bad_many_halogen_strong_negative.sdf` | 卤素数量过多 | 强负样本 | 512 | 总卤素或重卤素过多；一两个卤素可以接受，五六个通常应作为负样本。 |
| `primary_unique/04_bad_fused_heterocycle_strong_negative.sdf` | 稠合杂环/复杂杂环 | 强负样本 | 17142 | 杂环稠合过多或结构过复杂，常对应视觉上不合理的复杂环体系。 |
| `primary_unique/05_bad_poly_fused_or_bridge_ring_strong_negative.sdf` | 多稠合或桥环体系 | 强负样本 | 403 | 最大稠合环系统过大，或存在桥头/螺原子等复杂环拓扑。 |
| `primary_unique/06_bad_large_ring_strong_negative.sdf` | 七元及以上大环 | 强负样本 | 7782 | 存在七元及以上大环；尤其是大环又与六元环稠合时，作为当前生成坏模式强负样本。 |
| `primary_unique/07_bad_multi_aromatic_heterocycle_negative.sdf` | 多芳香杂环 | 负样本 | 11 | 芳香杂环数量偏多，可能造成 scaffold 过复杂或偏离目标分布。 |
| `primary_unique/08_bad_three_member_ring_strong_negative.sdf` | 三元环 | 强负样本 | 1209 | 三元环张力大；在当前生成结果中大量出现时通常是不合理模式。 |
| `primary_unique/09_bad_high_carbon_low_hetero_negative.sdf` | 高碳低杂原子 | 负样本 | 572 | 碳比例过高且杂原子过少，提示模型往疏水高碳方向漂移。 |
| `primary_unique/10_bad_fragmented_strong_negative.sdf` | 碎片化分子 | 强负样本 | 532 | 分子被拆成多个片段，不能作为合理 ligand。 |

## 按所有触发原因计数

| 文件 | 中文原因 | 强度 | 数量 |
|---|---:|---:|---:|
| `all_triggered_reasons/01_bad_saturated_polycycle_no_aromatic_strong_negative.sdf` | 无芳香环的多饱和碳环 | 强负样本 | 12675 |
| `all_triggered_reasons/02_bad_overhalogenated_small_aromatic_strong_negative.sdf` | 小芳香环过度卤代 | 强负样本 | 55 |
| `all_triggered_reasons/03_bad_many_halogen_strong_negative.sdf` | 卤素数量过多 | 强负样本 | 541 |
| `all_triggered_reasons/04_bad_fused_heterocycle_strong_negative.sdf` | 稠合杂环/复杂杂环 | 强负样本 | 17373 |
| `all_triggered_reasons/05_bad_poly_fused_or_bridge_ring_strong_negative.sdf` | 多稠合或桥环体系 | 强负样本 | 8861 |
| `all_triggered_reasons/06_bad_large_ring_strong_negative.sdf` | 七元及以上大环 | 强负样本 | 18856 |
| `all_triggered_reasons/07_bad_multi_aromatic_heterocycle_negative.sdf` | 多芳香杂环 | 负样本 | 169 |
| `all_triggered_reasons/08_bad_three_member_ring_strong_negative.sdf` | 三元环 | 强负样本 | 10232 |
| `all_triggered_reasons/09_bad_high_carbon_low_hetero_negative.sdf` | 高碳低杂原子 | 负样本 | 1866 |
| `all_triggered_reasons/10_bad_fragmented_strong_negative.sdf` | 碎片化分子 | 强负样本 | 532 |

## 注意

这些规则是为了训练分类器和 reward guard 的负样本，不等于化学上绝对禁止。
例如一两个卤素可以是合理分子；这里强扣的是过度卤代、小分子过度卤代和复杂环模式。
