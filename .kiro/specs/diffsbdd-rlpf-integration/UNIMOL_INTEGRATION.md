# Uni-Mol Integration for Manifold Discriminator

## 决策摘要

在判别器（ManifoldDiscriminatorReward）中，我们需要将生成的3D分子与数据库药物进行比较，以判断生成分子是否位于真实药物流形上。经过深入分析Uni-Mol代码和项目需求，我们决定：

**使用SMILES作为判别器的统一输入格式，利用Uni-Mol自动处理3D特征提取。**

## 问题分析

### 初始困惑

1. **生成分子是3D的** - DiffSBDD生成的分子带有3D坐标（结合构象）
2. **数据库药物格式不一** - 下载的药物数据库主要是SMILES或2D结构
3. **构象差异问题** - 如果对数据库药物进行LigPrep，得到的是能量最低态，而不是结合构象

### 核心矛盾

**生成分子（结合构象）vs 数据库药物（能量最低态）→ 两者不可直接比较！**

- 生成的分子：在蛋白口袋中的特定构象（binding pose）
- 数据库分子经LigPrep：自由状态下的能量最低态（global minimum）
- 如果把数据库药物对接到我们的蛋白：没有意义，因为它们本来就不是为这个靶点设计的

## 解决方案：使用SMILES + Uni-Mol

### 方案原理

判别器的目标是判断：**"这是否是一个合理的、类药的分子"**，而不是**"这是否是一个好的结合构象"**。

因此，我们应该关注分子的**化学结构本身**，而不是它在某个特定蛋白里的姿态。

### Uni-Mol工作原理

通过深度分析Uni-Mol源代码（`/home/md666/data/disk6/Uni-Mol-main`），我们发现：

#### 1. Uni-Mol接受SMILES输入

```python
# unimol_tools/unimol_tools/data/conformer.py
def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True):
    '''
    将SMILES字符串转换为3D坐标
    '''
    mol = Chem.MolFromSmiles(smi)  # 从SMILES创建分子
    mol = AllChem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # 生成3D构象
    AllChem.MMFFOptimizeMolecule(mol)  # 能量优化
    coordinates = mol.GetConformer().GetPositions()
    return atoms, coordinates
```

#### 2. Uni-Mol内部自动生成3D构象

Uni-Mol的处理流程：
```
SMILES字符串
    ↓
RDKit.EmbedMolecule() - 生成3D构象
    ↓
MMFF能量优化
    ↓
提取3D特征（坐标、距离矩阵、边类型）
    ↓
Transformer编码 → 512维表示
```

#### 3. Uni-Mol输出统一的高维表示

```python
# unimol_tools/unimol_tools/predictor.py
class UniMolRepr:
    def get_repr(self, data, return_atomic_reprs=False):
        """
        输入：SMILES列表
        输出：
        - 'cls_repr': [N, 512] 分子级别表示
        - 'atomic_reprs': [N, max_atoms, 512] 原子级别表示（可选）
        """
```

### 为什么这个方案有效？

1. **构象无关性** - 同一个分子可以有很多构象，判别器应该关注化学结构，而不是特定构象
2. **自动3D处理** - Uni-Mol内部会为每个SMILES生成一致的3D表示，保证公平比较
3. **数据一致性** - 生成分子和数据库药物都从SMILES开始，经过相同的处理流程
4. **简化流程** - 数据库药物通常以SMILES格式存储，无需额外的3D构象生成

## 实现方案

### 1. 分子转换流程

```python
# ddpo/utils/molecule_converter.py

def coords_to_smiles(x: torch.Tensor, h: torch.Tensor,
                     dataset_info: Dict) -> Optional[str]:
    """
    将生成的3D分子转换为SMILES
    
    Workflow:
        3D coordinates (x, h) → RDKit Mol → SMILES
    
    Args:
        x: [N, 3] 原子坐标
        h: [N, F] 原子类型one-hot编码
        dataset_info: 数据集信息（包含atom_decoder）
    
    Returns:
        Canonical SMILES字符串，用于Uni-Mol输入
    """
    # 1. 构建RDKit分子对象
    mol = coords_to_rdkit_mol(x, h, dataset_info)
    if mol is None:
        return None
    
    # 2. 转换为SMILES
    try:
        mol_copy = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol_copy)  # 移除立体化学
        mol_copy = Chem.RemoveHs(mol_copy)    # 移除氢原子
        return Chem.MolToSmiles(mol_copy)     # 规范化SMILES
    except:
        return None
```

### 2. 判别器特征提取

```python
# ddpo/rewards/manifold_discriminator.py

class ManifoldDiscriminatorReward(BaseReward):
    def __init__(self, discriminator_checkpoint: str,
                 unimol_model_path: str, ...):
        # 初始化Uni-Mol编码器
        from unimol_tools import UniMolRepr
        self.unimol_encoder = UniMolRepr(
            data_type='molecule',
            remove_hs=False,
            model_name='unimolv1'
        )
    
    def extract_features(self, smiles: str) -> torch.Tensor:
        """
        提取化学结构特征用于判别（不依赖3D构象）
        
        Args:
            smiles: SMILES字符串（从生成的3D分子转换而来，或数据库药物）
        
        Returns:
            拼接特征 [2560维]:
            - ECFP4指纹: 2048维（从SMILES生成，药效团特征）
            - Uni-Mol表示: 512维（Uni-Mol从SMILES自动生成3D并提取）
        
        注意：
            - 不包含PoseBusters统计，因为数据库药物没有对接构象
            - PoseBusters应该在Stage 1（PhysicsReward）中单独检查
        """
        # 1. ECFP4指纹（从SMILES生成）
        mol_2d = Chem.MolFromSmiles(smiles)
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol_2d, 2, nBits=2048)  # [2048]
        
        # 2. Uni-Mol表示（Uni-Mol内部自动处理3D）
        unimol_repr = self.unimol_encoder.get_repr([smiles])
        unimol_emb = unimol_repr['cls_repr'][0]  # [512]
        
        # 拼接所有特征
        features = torch.cat([
            torch.tensor(ecfp4, dtype=torch.float32),
            unimol_emb
        ])  # [2560]
        
        return features
```

### 3. 数据库药物处理

```python
# 数据库药物准备（训练判别器时）
def prepare_real_drug_dataset():
    """
    准备真实药物数据集用于判别器训练
    """
    # 从ChEMBL/ZINC下载药物SMILES
    database_smiles = load_chembl_drugs()  # ['CCO', 'c1ccccc1', ...]
    
    # 提取特征（只需要SMILES，不需要3D构象）
    # 1. ECFP4指纹
    ecfp4_features = [compute_ecfp4(smi) for smi in database_smiles]
    
    # 2. Uni-Mol表示（Uni-Mol会自动生成3D）
    unimol_features = unimol_encoder.get_repr(database_smiles)
    
    # 拼接特征用于判别器训练
    real_features = concatenate([ecfp4_features, unimol_features])  # [N, 2560]
    return real_features
```

## 完整数据流

### 训练阶段

```
┌─────────────────────────────────────────────────────────────┐
│                    判别器训练数据准备                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  真实药物（ChEMBL/ZINC）                                      │
│  SMILES: ['CCO', 'c1ccccc1', ...]                           │
│      ↓                                                       │
│  1. ECFP4: SMILES → 2048维指纹                               │
│  2. Uni-Mol: SMILES → 自动生成3D → 512维表示                 │
│      ↓                                                       │
│  拼接特征 [2560维] → 标签: 1 (真实)                          │
│                                                              │
│  生成分子（早期训练）                                         │
│  3D坐标 (x, h)                                               │
│      ↓                                                       │
│  coords_to_smiles() → SMILES                                 │
│      ↓                                                       │
│  1. ECFP4: SMILES → 2048维指纹                               │
│  2. Uni-Mol: SMILES → 自动生成3D → 512维表示                 │
│      ↓                                                       │
│  拼接特征 [2560维] → 标签: 0 (生成)                          │
│                                                              │
│      ↓                                                       │
│  训练判别器: Binary Classifier (2560 → 1)                    │
└─────────────────────────────────────────────────────────────┘
```

### RL训练阶段（三阶段系统）

```
┌─────────────────────────────────────────────────────────────┐
│                    RL训练中的奖励计算                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DiffSBDD生成3D分子                                          │
│  (x, h) - 结合构象                                           │
│      ↓                                                       │
│  Stage 1: PhysicsReward (PoseBusters) - 第一道关卡          │
│      检查3D物理合理性（18项核心检查）                         │
│      不通过 → 惩罚 -10.0，结束                               │
│      通过 → 继续                                             │
│      ↓                                                       │
│  coords_to_smiles() → SMILES                                 │
│      ↓                                                       │
│  Stage 2: ManifoldDiscriminatorReward - 第二道关卡          │
│      提取化学结构特征:                                        │
│      ├─ ECFP4: SMILES → 2048维                               │
│      └─ Uni-Mol: SMILES → 自动3D → 512维                     │
│      ↓                                                       │
│  拼接特征 [2560维] → 判别器                                  │
│      ↓                                                       │
│  prob(real) > 0.5 → 奖励 0（通过，继续Stage 3）             │
│  prob(real) ≤ 0.5 → 惩罚 -5.0（不通过，结束）                │
│      ↓                                                       │
│  Stage 3: 其他奖励计算（仅对通过前两阶段的分子）             │
│      - DockingReward                                         │
│      - DruglikenessReward                                    │
│      - ResidueInteractionReward                              │
│      - KLConstraintReward                                    │
└─────────────────────────────────────────────────────────────┘
```

## 关键优势

### 1. 避免构象偏差
- 生成分子和数据库药物都从SMILES开始
- Uni-Mol为所有分子生成一致的3D表示
- 判别器关注化学结构，而不是特定构象

### 2. 利用Uni-Mol的预训练知识
- Uni-Mol在209M分子3D构象上预训练
- 学习到了丰富的3D几何和化学知识
- 512维表示捕获了分子的高维特征

### 3. 特征互补性

- **ECFP4 (2048维)**: 药效团，局部结构模式，捕获官能团信息
- **Uni-Mol (512维)**: 3D几何，全局分子表示，捕获空间结构

**PoseBusters的位置**：不包含在判别器中，而是作为Stage 1（PhysicsReward）的独立检查，只对生成的3D分子进行物理合理性验证。

### 4. 简化数据处理
- 数据库药物直接使用SMILES，无需额外处理
- 生成分子只需一次转换：3D → SMILES
- 所有后续处理由Uni-Mol自动完成

## 实现注意事项

### 1. SMILES规范化
```python
# 确保SMILES的一致性
def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)  # 移除立体化学
    mol = Chem.RemoveHs(mol)         # 移除氢原子
    return Chem.MolToSmiles(mol)     # 规范化
```

### 2. Uni-Mol配置
```python
# 使用与预训练一致的配置
unimol_encoder = UniMolRepr(
    data_type='molecule',
    remove_hs=False,      # 保留氢原子信息
    model_name='unimolv1' # 或 'unimolv2'
)
```

### 3. 批处理优化
```python
# Uni-Mol支持批量处理
smiles_batch = [coords_to_smiles(x, h, info) for x, h in zip(xs, hs)]
unimol_reprs = unimol_encoder.get_repr(smiles_batch)  # 批量编码
```

### 4. 错误处理
```python
# 处理无效SMILES
def safe_extract_features(x, h, dataset_info):
    smiles = coords_to_smiles(x, h, dataset_info)
    if smiles is None:
        # 返回零向量或跳过
        return torch.zeros(2578)
    return extract_features(smiles, x.numpy())
```

## 参考资料

### Uni-Mol论文
- Zhou et al. (2023). "Uni-Mol: A Universal 3D Molecular Representation Learning Framework". ICLR 2023.
- 论文链接: https://openreview.net/forum?id=6K2RM6wVqKu

### Uni-Mol代码
- GitHub: https://github.com/deepmodeling/Uni-Mol
- 本地路径: `/home/md666/data/disk6/Uni-Mol-main`
- 关键文件:
  - `unimol_tools/unimol_tools/data/conformer.py` - 3D构象生成
  - `unimol_tools/unimol_tools/predictor.py` - 表示学习接口
  - `unimol_tools/unimol_tools/data/datahub.py` - 数据处理

### 相关文档
- `design.md` - 整体设计文档
- `requirements.md` - 需求文档
- `tasks.md` - 任务列表

## 更新日志

- 2026-03-04: 初始版本，记录Uni-Mol集成决策和SMILES转换方案
