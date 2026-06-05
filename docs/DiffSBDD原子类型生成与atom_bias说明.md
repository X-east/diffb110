# DiffSBDD 原子类型生成与 atom bias 简明说明

## 1. 一句话

DiffSBDD 生成的不是“已经有键的分子”，而是一批 ligand nodes。

每个 node 同时包含：

```text
x = 3D 坐标
h = 原子类型分数向量
```

最后 `h` 里哪个元素分数最高，这个 node 就会被判成哪个原子。

atom bias 的作用就是：在最终 `argmax` 之前，给某些元素的分数加一点全局偏置，从而改变整体元素比例。

## 2. DiffSBDD 原子类型是怎么生成的

当前条件生成流程可以简化成：

```text
口袋 pocket
  ↓
采样 ligand node 数量
  ↓
从高斯噪声开始生成 ligand x,h
  ↓
扩散模型逐步去噪
  ↓
得到每个 ligand node 的连续坐标 x 和连续原子分数 h
  ↓
可选：h_corrected = h / temperature + atom_bias
  ↓
argmax(h_corrected)
  ↓
得到离散 atom type
  ↓
OpenBabel / reconstruction 根据坐标和元素推断键
```

关键代码位置：

- ligand 数量采样：`DiffSBDD/lightning_modules.py::generate_ligands`
- 条件采样：`ConditionalDDPM.sample_given_pocket`
- 最终原子类型选择：`DiffSBDD/equivariant_diffusion/conditional_model.py::sample_p_xh_given_z0`

最终选择逻辑本质上是：

```python
h_lig = apply_atom_score_bias(h_lig, atom_score_bias, temperature=atom_temperature)
h_lig = one_hot(argmax(h_lig))
```

也就是说，DiffSBDD 最后不是按化学价态选原子，而是按模型给出的连续分数最大值选原子。

## 3. 原始分数 h 是什么

`h` 可以理解成“每个元素的竞争分数”。

例如某个生成点的原始分数可能像：

```text
C: 1.20
N: 1.05
O: 0.40
S: -0.10
F: -0.30
```

如果不加 bias：

```text
argmax = C
```

如果给 N 加 `+0.20`：

```text
C: 1.20
N: 1.25
O: 0.40
S: -0.10
F: -0.30

argmax = N
```

所以 atom bias 不是凭空生成新点，也不改坐标，只是改变每个点最终被判成哪类元素的概率倾向。

注意：这里的 `h` 不是严格 softmax probability。它是扩散模型生成出来的连续 atom feature score。真正决定原子类型的是相对大小，不是绝对数值。

## 4. 原始分数分布现在有没有

严格说：目前没有完整保存 raw score distribution。

已有的 SDF 只能看到 `argmax` 和 reconstruction 之后的元素分布，例如 C/N/O/S/F/P/Cl 的最终比例。

真正的 raw score distribution 应该在以下位置保存：

```text
DiffSBDD/equivariant_diffusion/conditional_model.py
sample_p_xh_given_z0()
在 apply_atom_score_bias() 之前保存 h_lig
```

建议以后专门记录：

```text
每类元素 score 的 mean/std/q10/q50/q90
argmax 前 top1 元素分布
top1-top2 margin
哪些元素经常排第二
加 bias 前后有多少 node 改变了 atom type
```

这些指标比 SDF 后处理后的元素比例更能说明模型内部到底偏向什么元素。

## 5. 当前可见的元素分布

checkpoint 中的训练集 atom prior：

| element | train fraction |
|---|---:|
| C | 0.671 |
| N | 0.117 |
| O | 0.170 |
| S | 0.011 |
| Cl | 0.006 |
| P | 0.011 |
| F | 0.013 |

GPU 版 baseline 1000 个分子生成后，SDF 中的最终元素分布为：

| element | baseline generated fraction |
|---|---:|
| C | 0.661 |
| N | 0.018 |
| O | 0.218 |
| S | 0.046 |
| F | 0.056 |
| Cl | 0.001 |
| P | 0.000 |
| Br | 0.00005 |

来源：[metrics.csv](../analysis/final_refine_ablation_20260602_n1000_gpu/metrics.csv)

这说明：

- C 总体并不离谱。
- N 明显偏低。
- O/S/F 偏高。
- P/Cl 在 baseline 里几乎不出现。

但要注意，这个表是 `argmax + reconstruction` 之后的最终元素比例，不是 raw `h` 分数分布。

## 6. atom bias 的公式

当前 prior bias 使用的是 log-ratio correction：

```text
bias(element) = strength * log(target_fraction / baseline_fraction)
```

然后用 `clamp` 截断：

```text
bias = max(-clamp, min(clamp, bias))
```

含义：

| 情况 | bias 方向 | 结果 |
|---|---:|---|
| 某元素生成太少 | 正 bias | 更容易被 argmax 选中 |
| 某元素生成太多 | 负 bias | 更难被 argmax 选中 |
| 偏差极大 | clamp 截断 | 避免单次修太猛 |

例如 N 在 baseline 中很低，而训练集 prior 中较高，所以 N 会得到正 bias。

## 7. 为什么这次 P/Cl 会过冲

这次使用的是统一 clamp：

```text
strength = 0.5
clamp = 0.5
```

因为 P/Cl 在 baseline 里接近 0，而训练集 prior 里不是 0，所以：

```text
log(target / baseline)
```

会变得很大，最后被截断到 `+0.5`。

这会导致很多原本接近 C/N/O 边界的点被硬推成 P 或 Cl。

结果是：

- P/Cl fraction 上升。
- sanitize rate 下降。
- bad bond length 上升。
- QED 下降。

所以统一 bias 对 rare elements 很危险。

## 8. 你能不能通过调这个分布影响全局

可以。

atom bias 本质上就是全局调元素倾向。

但它有三个限制：

1. 它调的是全局倾向，不是每个分子的精确元素个数。
2. 它只改 atom type，不改坐标，也不改键。
3. 它不知道某个位置放 N/P/Cl 是否满足价态和成键规则。

所以它适合做“轻微纠偏”，不适合强行把元素比例拉到某个目标。

## 9. 更安全的调节原则

当前不建议继续使用统一 `clamp=0.5`。

更稳的策略：

```text
C/N/O/S/F 可以小幅调。
P/Cl/Br/I/B 先不调，或只允许极小 bias。
```

建议初始方向：

| element | 建议 |
|---|---|
| C | 轻微负 bias 或不动 |
| N | 小正 bias |
| O | 小负 bias |
| S | 小负 bias |
| F | 小负 bias |
| P | 先固定 0 |
| Cl | 先固定 0 或极小正 bias |
| Br/I/B | 先固定 0 |

推荐先试：

```text
strength = 0.15 / 0.25 / 0.35
common element clamp = 0.20 / 0.25
rare element clamp = 0.00 / 0.05
```

## 10. 最重要的结论

atom bias 可以调全局元素分布，但它不是完整解决方案。

它能回答：

```text
模型整体是不是太爱生成 C/O/F，太少生成 N？
```

但它不能直接解决：

```text
键长不合理
乱三元环
价态异常
点云几何差
OpenBabel 成键错
```

所以 atom bias 应该作为轻量 atom prior 护栏使用；真正要改善化学结构，还需要 bond-aware reconstruction、bond predictor，或者后续 edge/bond diffusion。

