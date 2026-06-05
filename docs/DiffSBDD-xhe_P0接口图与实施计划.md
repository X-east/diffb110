# DiffSBDD-xhe P0 接口图与实施计划

> 最后更新：2026-06-04

## 1. P0 结论

本轮只做代码阅读和接口设计，不修改训练代码。结论是：

> DiffSBDD 现有数据加载层已经部分具备 halfedge 能力；真正缺口在 LightningModule 没把 halfedge 字段传给 `ConditionalDDPM`，以及 DDPM/dynamics/sampler 尚未支持 `e_t`。

因此下一步不是新建数据加载体系，而是沿用现有 DiffSBDD 数据流，新增可选 `xhe` 字段传递和模型分支。

## 2. 当前 x/h 数据流

```text
train.npz / val.npz
    ↓
ProcessedLigandPocketDataset
    ↓
collate_fn
    ↓
LigandPocketDDPM.get_ligand_and_pocket(data)
    ↓
ligand = {
    x,
    one_hot,
    size,
    mask
}
pocket = {
    x,
    one_hot,
    size,
    mask
}
    ↓
LigandPocketDDPM.forward(data)
    ↓
ConditionalDDPM.forward(ligand, pocket)
    ↓
xh0_lig = cat(ligand['x'], ligand['one_hot'])
xh0_pocket = cat(pocket['x'], pocket['one_hot'])
    ↓
noised_representation(xh0_lig, xh0_pocket)
    ↓
EGNNDynamics.forward(z_t_lig, xh_pocket, t, ligand_mask, pocket_mask)
    ↓
net_out_lig = eps_xh
    ↓
x/h diffusion loss
```

采样流：

```text
LigandPocketDDPM.generate_ligands(...)
    ↓
prepare_pocket(...)
    ↓
ddpm.sample_given_pocket(pocket, num_nodes_lig)
    ↓
sample initial z_lig = (x, h)
    ↓
for t -> s:
    sample_p_zs_given_zt(...)
        ↓
        dynamics(...)
        ↓
        update z_lig
    ↓
sample_p_xh_given_z0(...)
    ↓
reconstruct_molecule(...)
```

## 3. 已有 halfedge 数据能力

代码阅读发现 `DiffSBDD/dataset.py` 已经支持 bond/halfedge 字段：

### 3.1 原始 bond 字段

如果 `.npz` 中存在：

```text
lig_bond_index
lig_bond_type
lig_bond_mask
num_lig_bonds
```

`ProcessedLigandPocketDataset.__getitem__()` 会调用：

```python
build_ligand_halfedges(
    num_lig_atoms,
    lig_bond_index,
    lig_bond_type,
)
```

并生成：

```text
ligand_halfedge_index
ligand_halfedge_type
num_ligand_halfedges
```

### 3.2 collate 已支持 index offset

`collate_fn()` 已经特殊处理：

```text
lig_bond_index
ligand_halfedge_index
```

会按 batch 中 ligand atom offset 做 index 平移。

同时，如果存在 `ligand_halfedge_type`，会生成：

```text
ligand_halfedge_type_batch
```

这基本等价于 standalone edge 代码里的：

```text
edge_batch
```

### 3.3 关键缺口

当前 `LigandPocketDDPM.get_ligand_and_pocket()` 只传：

```python
ligand = {
    'x': data['lig_coords'],
    'one_hot': data['lig_one_hot'],
    'size': data['num_lig_atoms'],
    'mask': data['lig_mask'],
}
```

没有把以下字段放进 ligand：

```text
ligand_halfedge_index
ligand_halfedge_type
ligand_halfedge_type_batch
num_ligand_halfedges
```

所以 P1 的第一步很小：

> 在 `bond_diffusion.enabled=True` 时，把 halfedge 字段从 batch data 传入 ligand dict；旧模式不读、不要求这些字段。

## 4. 当前 DDPM / dynamics 缺口

### 4.1 ConditionalDDPM

当前 `ConditionalDDPM.forward()` 只处理：

```python
xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
z_t_lig, xh_pocket, eps_t_lig = noised_representation(...)
net_out_lig, _ = dynamics(...)
```

缺少：

```text
e_0 = ligand['ligand_halfedge_type']
e_t = q(e_t | e_0)
pred_e0_logits = dynamics(..., e_t)
edge loss = KL/CE(pred_e0_logits, e_0)
```

### 4.2 EGNNDynamics

当前 `EGNNDynamics.forward()` 签名：

```python
forward(xh_atoms, xh_residues, t, mask_atoms, mask_residues)
```

它内部已有“边类型 embedding”，但这个 edge type 只是图结构类别：

| id | 当前含义 |
|---:|---|
| 0 | ligand-pocket |
| 1 | ligand-ligand |
| 2 | pocket-pocket |

这不是化学 bond type。

DiffSBDD-xhe 需要新增 ligand-ligand chemical halfedge：

| id | bond type |
|---:|---|
| 0 | no bond |
| 1 | single |
| 2 | double |
| 3 | triple |
| 4 | aromatic |

因此不能直接复用现有 `edge_embedding(3)` 当 bond diffusion，需要新增：

```text
chemical_edge_embedding(num_edge_types=5)
edge_decoder_head(...)
```

## 5. DiffSBDD-xhe 目标数据流

```text
train.npz / val.npz
    ↓
ProcessedLigandPocketDataset
    ↓
collate_fn
    ↓
data contains:
    lig_coords
    lig_one_hot
    lig_mask
    ligand_halfedge_index
    ligand_halfedge_type
    ligand_halfedge_type_batch
    num_ligand_halfedges
    ↓
LigandPocketDDPM.get_ligand_and_pocket(data)
    ↓
ligand = {
    x,
    one_hot,
    size,
    mask,
    halfedge_index,
    halfedge_type,
    edge_batch
}
    ↓
ConditionalDDPMXHE.forward(ligand, pocket)
    ↓
xh0_lig = cat(x, one_hot)
e0_lig = halfedge_type
    ↓
z_t_lig = q(xh_t | xh0)
e_t_lig = q(e_t | e0)
    ↓
EGNNDynamicsXHE.forward(..., halfedge_index, e_t, edge_batch)
    ↓
net_out_lig = eps_xh
pred_edge_logits = logits_e0
    ↓
loss = old_xh_loss + edge_loss + bond_length_loss
```

## 6. 建议新增/修改模块

### 6.1 最小新增模块

| 文件 | 动作 | 原因 |
|---|---|---|
| `DiffSBDD/equivariant_diffusion/xhe_schema.py` | 新增轻量 dataclass | 避免到处传 tuple |
| `DiffSBDD/equivariant_diffusion/edge_transition.py` | 复用/迁移 `EdgeCategoricalTransition` | 放到模型内部可依赖位置 |
| `DiffSBDD/equivariant_diffusion/dynamics_xhe.py` | 新增 `EGNNDynamicsXHE` | 不破坏原 `EGNNDynamics` |
| `DiffSBDD/equivariant_diffusion/conditional_model_xhe.py` | 新增 `ConditionalDDPMXHE` | 不破坏原 `ConditionalDDPM` |

### 6.2 必须小改的现有模块

| 文件 | 修改 |
|---|---|
| `DiffSBDD/lightning_modules.py` | 增加 `diffusion_mode` / `bond_diffusion` 配置分支；传递 halfedge 字段 |
| `DiffSBDD/reconstruction/builder.py` | 已支持 predicted_edge，后续只需接 sampler 输出 |
| `DiffSBDD/generate_ligands.py` | 最后阶段才增加 xhe predicted-edge 输出选项 |

### 6.3 暂时不能碰

| 模块 | 原因 |
|---|---|
| `ddpo/` | 当前 RL 继续使用原 DiffSBDD x/h |
| `launch/smoke.sh` | 旧 RL smoke 不能被 xhe 影响 |
| `ddpo/integration/diffsbdd_model.py` | xhe supervised 稳定前不接 RL |

## 7. 配置设计

旧模式默认：

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

兼容规则：

1. `diffusion_mode=xh` 时，不要求数据里有 halfedge 字段。
2. `diffusion_mode=xhe` 时，必须检查 halfedge 字段存在。
3. 旧 checkpoint 加载不能失败；新 edge 参数允许 missing。
4. 旧 generation/RL 路径默认仍使用 OpenBabel/EDM reconstruction。

## 8. P1 最小实现建议

下一步先不要直接改完整 sampler。先做最小可测闭环：

### P1-A：传字段

在 `LigandPocketDDPM.get_ligand_and_pocket()` 增加：

```python
if self.bond_diffusion_enabled:
    ligand['halfedge_index'] = data['ligand_halfedge_index'].to(self.device, INT_TYPE)
    ligand['halfedge_type'] = data['ligand_halfedge_type'].to(self.device, INT_TYPE)
    ligand['edge_batch'] = data['ligand_halfedge_type_batch'].to(self.device, INT_TYPE)
```

测试：

- xh 模式旧数据可 forward。
- xhe 模式缺 halfedge 字段时报清楚错误。
- xhe 模式有字段时 shape 正确。

### P1-B：edge-only head smoke

先不改 `sample_p_zs_given_zt()`。

只做：

```text
真实 x/h noising
真实 e noising
        ↓
dynamics 输出 pred_edge_logits
        ↓
edge CE/KL loss
```

目标：

- 证明 DiffSBDD 内部 hidden state 能预测 bond type。
- 对照 standalone 10k DDP 的 `real-bond F1≈0.85`。

### P1-C：warm-start 检查

加载原始 DiffSBDD checkpoint：

```text
strict=False
missing keys = edge embedding/head
unexpected keys = 0 或可解释
```

旧模式：

```text
diffusion_mode=xh
```

必须仍能生成和 RL smoke。

## 9. 验证标准

| 阶段 | 标准 |
|---|---|
| P1 字段传递 | shape 对齐，旧模式不受影响 |
| P2 edge head | 10k eval real-bond F1 明显上升 |
| P3 joint loss | x/h loss 不崩，edge loss 下降 |
| P4 sampler | reconstruct sanitize 接近 standalone 0.78 |
| P5 prior | prior sanitize 从 0 抬头 |
| P6 RL 前 | 旧 `launch/smoke.sh` 通过 |

## 10. 最重要的工程边界

DiffSBDD-xhe 必须是可选模式：

```text
默认 xh:
    当前 DiffSBDD + RL 完全不变

启用 xhe:
    只在专门训练/生成配置中生效
```

任何实现如果导致旧 checkpoint、旧生成、旧 DDPO smoke 受影响，都应该回退。
