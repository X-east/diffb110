# DiffSBDD-xhe 改造方案

> 最后更新：2026-06-04

## 1. 核心结论

本项目的大改线最终目标不是另起炉灶训练一个全新的独立 x/h/e 模型，而是把当前 DiffSBDD 本体改造成**可选的 x/h/e 扩散模式**：

```text
DiffSBDD 当前模式:
    z_t = (x_t, h_t)

DiffSBDD-xhe 新模式:
    z_t = (x_t, h_t, e_t)
```

其中：

| 符号 | 含义 |
|---|---|
| `x` | ligand 原子三维坐标 |
| `h` | ligand 原子类型 |
| `e` | ligand halfedge bond type |

因此，大改线的正确定位是：

> 保留 DiffSBDD 的 protein-conditioned 生成主链、size distribution、采样入口、RL 兼容边界，在其内部加入 edge diffusion，而不是把已有 DiffSBDD 丢掉。

## 2. standalone edge diffusion 的定位

近期新增的 `DiffSBDD/edge_diffusion/`、`train_edge_diffusion_ddp.py` 和 sampler smoke 不是最终底座模型。它们的定位是隔离验证工具：

| 已验证内容 | 价值 |
|---|---|
| halfedge 数据链 | 确认 `ligand_halfedge_index/type` 可从 CrossDocked 构建 |
| oracle / semantic reconstruction | 确认真实 bond type 能反构建分子 |
| categorical edge transition | 确认离散 bond diffusion 数学接口可用 |
| x/h/e 最小 denoiser | 确认 bond denoising 是可学习任务 |
| DDP 训练入口 | 确认可多卡训练 10k edge 数据 |
| predicted-edge reconstruction | 确认不用 OpenBabel 也能从 predicted bond type 建分子 |

它不能直接替代 DiffSBDD，因为它缺少：

- DiffSBDD 原有 protein-conditioned dynamics。
- DiffSBDD ligand size distribution。
- DiffSBDD 现有采样链和 `generate_ligands` 接口。
- 当前 DDPO/RL 需要的 DiffSBDD logprob / rollout 兼容边界。
- 真实 de novo prior generation 所需的完整连续坐标 posterior 和 atom transition。

## 3. 当前实验怎么解释

10k / 6GPU / 3 epoch edge DDP 测试结果：

| 指标 | 值 |
|---|---:|
| train / eval molecules | 9000 / 1000 |
| epoch | 3 |
| eval real-bond F1 | 0.8516 |
| eval atom accuracy | 0.9897 |
| eval pos RMSE | 0.0242 |
| reconstruct t100 sanitize | 0.7813 |
| reconstruct bad bond length | 0 |
| prior t100 sanitize | 0 |

解释：

1. `eval real-bond F1=0.85` 说明 bond denoising 任务本身可学。
2. `reconstruct sanitize=0.78` 说明真实分子附近的 noisy state 可以被还原。
3. `prior sanitize=0` 说明 standalone 小模型不能从随机 x/h/e prior 生成分子。
4. 这不证明 DiffSBDD-xhe 不可行，只证明 standalone 小模型不是最终生成底座。

因此下一步不是继续把 standalone 小模型打磨成新 base，而是把已经验证过的数据链、transition、edge head、predicted-edge reconstruction 迁移进 DiffSBDD 本体。

## 4. DiffSBDD 现有接入点

当前 DiffSBDD 核心链路：

```text
LigandPocketDDPM
    -> ConditionalDDPM
        -> EGNNDynamics
            -> EGNN/GNN dynamics
```

关键现状：

- `ConditionalDDPM.forward()` 中当前构造：

```python
xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
```

- `EGNNDynamics.forward()` 当前输入：

```python
forward(xh_atoms, xh_residues, t, mask_atoms, mask_residues)
```

- `sample_given_pocket()` 当前从 Gaussian prior 初始化 `z_lig = (x, h)`，再逐步调用：

```python
sample_p_zs_given_zt(...)
```

- 分子重建已经支持：

```text
reconstruct_molecule(..., method="predicted_edge")
```

这意味着 DiffSBDD-xhe 的最小侵入接入点是：

| 层级 | 当前 | xhe 改造 |
|---|---|---|
| dataset | ligand x / one_hot / mask | 增加 halfedge_index / halfedge_type |
| diffusion state | `z_t = xh_t` | `z_t = (xh_t, e_t)` |
| dynamics input | xh + pocket + t | xh + pocket + t + e_t |
| dynamics output | eps_xh | eps_xh + pred_edge_logits |
| loss | x/h diffusion loss | x/h loss + edge categorical loss + bond length loss |
| sampler | sample x/h | sample x/h/e |
| reconstruction | OpenBabel/EDM | predicted_edge 可选 |

## 5. 目标架构

### 5.1 配置开关

必须保持原始 DiffSBDD 默认行为不变：

```yaml
model:
  diffusion_mode: xh
  bond_diffusion:
    enabled: false
```

新模式：

```yaml
model:
  diffusion_mode: xhe
  bond_diffusion:
    enabled: true
    num_edge_types: 5
    edge_prior: uniform
    edge_loss_weight: 1.0
    bond_length_loss_weight: 0.5
```

要求：

- `enabled=false` 时，所有旧 checkpoint、旧采样、旧 RL 继续可用。
- `enabled=true` 时，才读取 halfedge 字段、构建 edge transition、输出 edge logits。

### 5.2 数据接口

在现有 ligand batch 中增加：

```python
ligand['halfedge_index']  # [2, E]
ligand['halfedge_type']   # [E]
ligand['edge_batch']      # [E]
```

这部分可以复用 standalone 数据链中的：

- `DiffSBDD/edge_utils.py`
- `EdgeBundleDataset`
- `EdgeBatchCollator`
- 10k edge data chain 构建报告

但最终训练数据应该接入 DiffSBDD 原训练数据加载链，而不是长期依赖 standalone `EdgeBundleDataset`。

### 5.3 dynamics 改造

不应直接替换 `EGNNDynamics`，而是新增可选包装：

```text
EGNNDynamics          # 原始 x/h dynamics，默认继续使用
EGNNDynamicsXHE       # 新增，支持 e_t 输入和 edge logits 输出
```

`EGNNDynamicsXHE` 可以复用：

- 原 atom/residue encoder。
- 原 EGNN message passing。
- 原 coordinate / atom decoder。

新增：

- ligand-ligand halfedge embedding。
- timestep-conditioned edge embedding。
- edge decoder head：

```python
pred_edge_logits = edge_head(node_h_i, node_h_j, edge_h_ij, distance_ij, t)
```

### 5.4 diffusion 改造

不建议把现有 `ConditionalDDPM` 直接改烂。应新增子类或模式分支：

```text
ConditionalDDPM       # 原始 x/h
ConditionalDDPMXHE    # 新 x/h/e
```

`ConditionalDDPMXHE` 复用原逻辑：

- normalize / unnormalize
- center-of-mass 处理
- gamma schedule
- x/h Gaussian diffusion
- sample_p_zs_given_zt
- sample_given_pocket 主循环

新增：

- `EdgeCategoricalTransition`
- `q(e_t | e_0)`
- `q(e_{t-1} | e_t, pred_e0)`
- edge KL/CE loss
- predicted-edge reconstruction 输出

### 5.5 checkpoint warm-start

不要从零训练整个 DiffSBDD-xhe。

推荐：

```text
加载原始 DiffSBDD pretrained checkpoint
        ↓
复用 x/h dynamics 权重
        ↓
新增 edge embedding / edge head / edge transition
        ↓
先训练 edge 相关模块
        ↓
再 joint finetune x/h/e
```

阶段建议：

| 阶段 | 训练策略 |
|---|---|
| A | 冻结大部分原 DiffSBDD，只训练 edge head / edge embedding |
| B | 解冻 dynamics 后几层，联合训练 x/h/e |
| C | 全模型小学习率 finetune |
| D | sampler/reconstruction 稳定后再考虑 RL |

## 6. 与强化学习的边界

当前不能把 xhe 直接接入 RL。

原因：

- prior generation 尚未稳定。
- edge diffusion 会改变 logprob 定义。
- PPO/DDPO 需要完整 trajectory logprob，`e_t` 也必须进入 logprob。
- 如果 prior sanitize 仍为 0，RL 会放大奖励漏洞。

正确顺序：

```text
DiffSBDD-xhe supervised pretrain
        ↓
reconstruct / prior sampler smoke
        ↓
predicted-edge molecule quality benchmark
        ↓
再设计 x/h/e trajectory logprob
        ↓
最后接 DDPO/RL
```

原始 RL 主线继续使用：

```yaml
model:
  diffusion_mode: xh
```

## 7. 分阶段实施计划

### P0：代码阅读和接口图

目标：明确 DiffSBDD-xhe 改哪些点，不动代码。

检查文件：

- `DiffSBDD/equivariant_diffusion/conditional_model.py`
- `DiffSBDD/equivariant_diffusion/dynamics.py`
- `DiffSBDD/lightning_modules.py`
- `DiffSBDD/generate_ligands.py`
- `DiffSBDD/reconstruction/builder.py`
- `ddpo/integration/diffsbdd_model.py`

输出：

- 接口图。
- 需要新增字段列表。
- 旧模式兼容性清单。

### P1：数据字段接入

目标：DiffSBDD 原数据 batch 可以携带 halfedge 字段。

成功标准：

- 旧数据 batch 不受影响。
- 新数据 batch 有 `halfedge_index/type/edge_batch`。
- 单元测试确认 shape 和 molecule alignment。

### P2：模型 forward smoke

目标：在 DiffSBDD dynamics 中增加 edge head，但不改 sampler。

成功标准：

- `diffusion_mode=xh` 行为不变。
- `diffusion_mode=xhe` 能 forward。
- edge logits shape 为 `[E, 5]`。
- 可对真实 edge type 计算 CE/KL。

### P3：训练 loss

目标：在 DiffSBDD training step 中加入 edge loss。

成功标准：

- 旧 checkpoint 可加载。
- x/h loss 正常。
- edge loss 下降。
- 10k train/val 上 real-bond F1 接近 standalone DDP 的 0.85。

### P4：sampler

目标：`sample_given_pocket()` 同时采样 x/h/e。

成功标准：

- reconstruct t100 sanitize 接近 standalone 或更好。
- prior sanitize 不再为 0。
- predicted-edge reconstruction 输出 SDF。

### P5：RL 前置检查

目标：确认 xhe base 能生成合理分子，再设计 RL 接口。

成功标准：

- 生成分子 validity/sanitize、bad bond length、atom composition、ring 分布达标。
- x/h/e logprob 定义清楚。
- 旧 xh RL smoke 仍通过。

## 8. 风险

| 风险 | 原因 | 对策 |
|---|---|---|
| 破坏现有 DiffSBDD/RL | 直接改原类 | 使用 `diffusion_mode` 开关，旧模式默认 |
| edge 类别极不均衡 | no-bond 约 91.7% | class weight / focal loss / existence+type 分解 |
| prior generation 仍失败 | sampler posterior 粗糙 | 对齐 DiffGui categorical posterior 和坐标 posterior |
| 速度下降 | 完整 halfedge 数量为 O(N²) | 半径裁剪 / top-k candidate edge / batch size 调整 |
| checkpoint 不兼容 | 新增参数 | strict=False warm-start，记录 missing/unexpected keys |
| RL logprob 不一致 | e_t 未纳入 trajectory | RL 接入前先定义 x/h/e joint logprob |

## 9. 当前推荐路线

短期不要继续把 standalone edge 模型当新 base 训练，而应转向：

```text
standalone 验证结果
        ↓
迁移到 DiffSBDD-xhe optional mode
        ↓
warm-start 原 DiffSBDD checkpoint
        ↓
10k/全量 CrossDocked supervised finetune
        ↓
predicted-edge generation benchmark
        ↓
再考虑 DDPO/RL
```

最重要的原则：

> 旧 DiffSBDD x/h 路线必须始终可运行；x/h/e 是可选新模式，不是强行替换。
