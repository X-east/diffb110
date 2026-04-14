# DiffSBDD: 基于结构的药物设计等变扩散模型

**DiffSBDD** 官方实现，一个用于基于结构的药物设计的等变扩散模型。

作者：Arne Schneuing, Charles Harris, Yuanqi Du, Kieran Didi, Arian Jamasb, Ilia Igashov, Weitao Du, Carla Gomes, Tom Blundell, Pietro Lio, Max Welling, Michael Bronstein & Bruno Correia.

[![DOI](https://zenodo.org/badge/DOI/10.1038/s43588-024-00737-x.svg)](https://doi.org/10.1038/s43588-024-00737-x)
[![arXiv](https://img.shields.io/badge/arXiv-2210.13695-B31B1B.svg)](http://arxiv.org/abs/2210.13695)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arneschneuing/DiffSBDD/blob/main/colab/DiffSBDD.ipynb)

> [!TIP]
> 您也可以在 https://github.com/LPDI-EPFL/DrugFlow 尝试我们新的 3D 药物设计生成模型。

![](img/overview.png)

## 目录

1. [依赖项](#依赖项)
   1. [Conda 环境](#conda-环境)
   2. [预训练模型](#预训练模型)
2. [分步示例](#分步示例)
   1. [从头设计](#从头设计)
   2. [子结构修补](#子结构修补)
   3. [分子优化](#分子优化)
3. [基准测试](#基准测试)
   1. [CrossDocked 基准](#crossdocked)
   2. [Binding MOAD](#binding-moad)
   3. [采样分子](#采样分子)
4. [训练](#训练)
5. [推理](#推理)
   1. [为给定口袋采样分子](#为给定口袋采样分子)
   2. [测试集采样](#为测试集中所有口袋采样分子)
   3. [固定子结构](#固定子结构)
   4. [评估指标](#评估指标)
6. [引用](#引用)

---

## 依赖项

### Conda 环境

可以使用以下命令创建并激活 conda 环境：
```bash
conda env create -f environment.yaml -n diffsbdd
conda activate diffsbdd
```

<details>
<summary>旧版安装说明</summary>

```bash
conda create -n sbdd-env
conda activate sbdd-env
conda install pytorch cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pytorch-lightning
conda install -c conda-forge wandb
conda install -c conda-forge rdkit
conda install -c conda-forge biopython
conda install -c conda-forge imageio
conda install -c anaconda scipy
conda install -c pyg pytorch-scatter
conda install -c conda-forge openbabel
conda install seaborn
```

代码测试时使用的版本：
| 软件              | 版本      |
|-------------------|-----------|
| Python            | 3.10.4    |
| CUDA              | 10.2.89   |
| PyTorch           | 1.12.1    |
| PyTorch Lightning | 1.7.4     |
| WandB             | 0.13.1    |
| RDKit             | 2022.03.2 |
| BioPython         | 1.79      |
| imageio           | 2.21.2    |
| SciPy             | 1.7.3     |
| PyTorch Scatter   | 2.0.9     |
| OpenBabel         | 3.1.1     |
</details>

### 预训练模型

预训练模型可从 [Zenodo](https://zenodo.org/record/8183747) 下载：

- [CrossDocked, 条件 Cα 模型](https://zenodo.org/record/8183747/files/crossdocked_ca_cond.ckpt?download=1)
- [CrossDocked, 联合 Cα 模型](https://zenodo.org/record/8183747/files/crossdocked_ca_joint.ckpt?download=1)
- [CrossDocked, 条件全原子模型](https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt?download=1)
- [CrossDocked, 联合全原子模型](https://zenodo.org/record/8183747/files/crossdocked_fullatom_joint.ckpt?download=1)
- [Binding MOAD, 条件 Cα 模型](https://zenodo.org/record/8183747/files/moad_ca_cond.ckpt?download=1)
- [Binding MOAD, 联合 Cα 模型](https://zenodo.org/record/8183747/files/moad_ca_joint.ckpt?download=1)
- [Binding MOAD, 条件全原子模型](https://zenodo.org/record/8183747/files/moad_fullatom_cond.ckpt?download=1)
- [Binding MOAD, 联合全原子模型](https://zenodo.org/record/8183747/files/moad_fullatom_joint.ckpt?download=1)

---

## 分步示例

这些简单的分步示例提供了使用 DiffSBDD 生成分子的入门指南。
下面提供了更多关于训练和采样脚本的详细信息。

在运行采样脚本之前，我们需要下载模型检查点：
```bash
wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt
```
它将存储在 `./checkpoints` 文件夹中。

### 从头设计

使用训练好的模型权重，我们可以用一条命令采样新的配体。在这个例子中，我们使用 PDB ID 为 `3RFM` 的蛋白质，可以在 example 文件夹中找到。
PDB 文件包含一个参考配体，位于链 A 的残基编号 330，我们可以用它来指定目标结合口袋。
以下命令将生成 20 个样本，并将它们保存在 `./example` 文件夹中名为 `3rfm_mol.sdf` 的文件中。

```bash
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand A:330 --n_samples 20
```

除了指定链和残基编号，我们也可以提供包含参考配体的 SDF 文件：
```bash
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand example/3rfm_B_CFF.sdf --n_samples 20
```

如果没有已知的参考配体，结合口袋也可以指定为残基列表，如[下文](#为给定口袋采样分子)所述。

### 子结构修补

要围绕固定子结构设计分子（骨架扩展、片段连接等），可以运行 `inpaint.py` 脚本。
这里，我们用一个片段连接的例子来演示其用法。与 `generate_ligands.py` 类似，修补脚本允许我们基于 SDF 格式的参考配体或使用链和残基标识符（如果它在 PDB 中）来定义口袋。

固定子结构最简单的方法是使用 `--fix_atoms` 标志在单独的 SDF 文件中提供它们。
但是，脚本也接受原子名称列表，这些名称必须对应于 PDB 文件中参考配体的原子，例如 `--fix_atoms C1 N6 C5 C12`。

```bash 
python inpaint.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/5ndu.pdb --outfile example/5ndu_linked_mols.sdf --ref_ligand example/5ndu_C_8V2.sdf --fix_atoms example/fragments.sdf --center ligand --add_n_nodes 10
```

注意 `--center ligand` 选项告诉 DiffSBDD 在固定子结构的质心附近采样额外的原子，这并不总是理想或期望的。
例如，输入可能是两个大小非常不同的片段，在这种情况下，随机噪声将在较大片段附近采样。
我们目前也支持在口袋中心采样（`--center pocket`），但在某些情况下，这两个选项都可能不合适，需要针对特定问题的解决方案以避免不良结果。

另一个重要参数是 `--add_n_nodes`，它决定将添加多少个新原子。如果未提供，将随机采样一个数字。

### 分子优化

您可以使用 DiffSBDD 通过 `optimize.py` 脚本针对给定属性优化现有分子。

```bash 
python optimize.py --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/5ndu.pdb --outfile output.sdf --ref_ligand example/5ndu_C_8V2.sdf --objective sa --population_size 100 --evolution_steps 10 --top_k 10 --timesteps 100
```

进化算法中的重要参数：
- `--checkpoint`: 用于加噪-去噪模型的检查点。
- `--objective`: 优化目标。目前支持 'qed'（药物相似性定量估计）和 'sa'（合成可及性）。可以在代码中实现自定义目标。
- `--population_size`: 在优化代际中维护的分子种群大小。
- `--evolution_steps`: 优化过程中执行的进化步数（代数）。
- `--top_k`: 从一代到下一代选择的得分最高的分子数量。
- `--timesteps`: 优化算法中使用的加噪-去噪步数。默认为 100（总共 T=500）。

---

## 基准测试

### CrossDocked

#### 数据准备
按照 Pocket2Mol 作者的描述下载并解压数据集：https://github.com/pengxingang/Pocket2Mol/tree/main/data

使用以下命令处理原始数据：
```bash
python process_crossdock.py <crossdocked_dir> --no_H
```

### Binding MOAD

#### 数据准备
下载数据集：
```bash
wget http://www.bindingmoad.org/files/biou/every_part_a.zip
wget http://www.bindingmoad.org/files/biou/every_part_b.zip
wget http://www.bindingmoad.org/files/csv/every.csv

unzip every_part_a.zip
unzip every_part_b.zip
```

使用以下命令处理原始数据：
```bash
python -W ignore process_bindingmoad.py <bindingmoad_dir>
```
添加 `--ca_only` 标志以创建使用 Cα 口袋表示的数据集。

### 采样分子
采样的分子可以在 [Zenodo](https://zenodo.org/record/8239058) 上找到。

---

## 训练

开始新的训练运行：
```bash
python -u train.py --config <config>.yml
```

恢复之前的运行：
```bash
python -u train.py --config <config>.yml --resume <checkpoint>.ckpt
```

---

## 推理

### 为给定口袋采样分子

使用训练好的模型为给定口袋采样小分子，使用以下命令：
```bash
python generate_ligands.py <checkpoint>.ckpt --pdbfile <pdb_file>.pdb --outfile <output_file> --resi_list <口袋残基ID列表>
```

例如：
```bash
python generate_ligands.py last.ckpt --pdbfile 1abc.pdb --outfile results/1abc_mols.sdf --resi_list A:1 A:2 A:3 A:4 A:5 A:6 A:7 
```

或者，结合口袋也可以基于同一 PDB 文件中的参考配体指定：
```bash 
python generate_ligands.py <checkpoint>.ckpt --pdbfile <pdb_file>.pdb --outfile <output_file> --ref_ligand <chain>:<resi>
```

或使用单独的 SDF 文件：
```bash 
python generate_ligands.py <checkpoint>.ckpt --pdbfile <pdb_file>.pdb --outfile <output_file> --ref_ligand <ref_ligand>.sdf
```

可选标志：
| 标志 | 描述 |
|------|-------------|
| `--n_samples` | 采样分子数量 |
| `--num_nodes_lig` | 采样分子的大小（原子数） |
| `--timesteps` | 推理时的去噪步数 |
| `--all_frags` | 保留所有断开的片段 |
| `--sanitize` | 清理分子（如果存在此标志，无效分子将被移除） |
| `--relax` | 在力场中松弛生成的结构（不考虑蛋白质，可能引入冲突） |
| `--resamplings` | 修补参数（如果使用条件模型则不适用） |
| `--jump_length` | 修补参数（如果使用条件模型则不适用） |

### 为测试集中所有口袋采样分子

`test.py` 可用于为整个测试集采样分子：
```bash
python test.py <checkpoint>.ckpt --test_dir <bindingmoad_dir>/processed_noH/test/ --outdir <output_dir> --sanitize
```

有不同的方法来确定采样分子的大小：
- `--fix_n_nodes`: 生成与参考分子具有相同节点数的配体
- `--n_nodes_bias <int>`: 随机采样节点数并添加此偏差
- `--n_nodes_min <int>`: 随机采样节点数但将其限制在此值

其他可选标志与 `generate_ligands.py` 类似。

### 固定子结构

`inpaint.py` 可用于使用条件训练模型进行部分配体重新设计，例如：
```bash 
python inpaint.py <checkpoint>.ckpt --pdbfile <pdb_file>.pdb --outfile <output_file> --ref_ligand <chain>:<resi> --fix_atoms C1 N6 C5 C12
```
`--add_n_nodes` 控制新生成节点的数量。其他选项与之前相同。

### 评估指标

要评估基本分子属性，创建 `MoleculeProperties` 类的实例并运行其 `evaluate` 方法：
```python
from analysis.metrics import MoleculeProperties
mol_metrics = MoleculeProperties()
all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = \
    mol_metrics.evaluate(pocket_mols)
```
`evaluate()` 期望一个列表的列表，其中内部列表包含为一个口袋生成的所有 RDKit 分子。

---

## 引用

```
@article{schneuing2024diffsbdd,
   title={Structure-based drug design with equivariant diffusion models},
   author={Schneuing, Arne and Harris, Charles and Du, Yuanqi and Didi, Kieran and Jamasb, Arian and Igashov, Ilia and Du, Weitao and Gomes, Carla and Blundell, Tom L and Lio, Pietro and Welling, Max and Bronstein, Michael and Correia, Bruno},
   journal={Nature Computational Science},
   year={2024},
   month={Dec},
   day={01},
   volume={4},
   number={12},
   pages={899-909},
   issn={2662-8457},
   doi={10.1038/s43588-024-00737-x},
   url={https://doi.org/10.1038/s43588-024-00737-x}
}
```

---

## 术语表（面向非计算机专业）

| 英文术语 | 中文 | 解释 |
|---------|------|------|
| Equivariant | 等变 | 模型对旋转/平移保持一致性 |
| Diffusion Model | 扩散模型 | 通过逐步去噪生成数据的模型 |
| Pocket | 口袋/结合位点 | 蛋白质上配体结合的凹陷区域 |
| Ligand | 配体 | 与蛋白质结合的小分子 |
| Checkpoint | 检查点 | 保存的模型权重文件 |
| Sampling | 采样 | 从模型生成新分子 |
| Inpainting | 修补 | 保留部分结构，生成其余部分 |
| De novo | 从头 | 完全新设计，不基于已有分子 |
| SDF | 结构数据文件 | 存储分子 3D 结构的文件格式 |
| PDB | 蛋白质数据库文件 | 存储蛋白质 3D 结构的文件格式 |
| QED | 药物相似性 | 衡量分子像药物的程度 (0-1) |
| SA | 合成可及性 | 衡量分子合成难度 (1-10，越小越易合成) |
| Cα | α碳 | 氨基酸主链上的碳原子 |
| Full-atom | 全原子 | 包含所有原子（不只是 Cα） |
| Conditional | 条件 | 基于口袋条件生成配体 |
| Joint | 联合 | 同时生成口袋和配体 |