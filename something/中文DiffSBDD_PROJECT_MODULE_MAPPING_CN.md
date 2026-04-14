# DiffSBDD 项目 - 全面模块与功能路径映射

**项目名称：** DiffSBDD (基于等变扩散模型的结构导向药物设计)  
**位置：** `DiffSBDD/`  
**用途：** 用于结构导向药物设计的等变扩散模型，支持 3D 分子生成

**论文：** Structure-based drug design with equivariant diffusion models (Nature Computational Science, 2024)  
**DOI:** [10.1038/s43588-024-00737-x](https://doi.org/10.1038/s43588-024-00737-x)  
**arXiv:** [2210.13695](http://arxiv.org/abs/2210.13695)

---

## 目录
1. [项目概览](#项目概览)
2. [核心架构](#核心架构)
3. [模块详情](#模块详情)
4. [统计摘要](#统计摘要)

---

## 项目概览

DiffSBDD 实现了用于结构导向药物设计的等变扩散模型。该模型在蛋白质结合口袋的条件下生成 3D 分子结构，支持：
- 从头配体设计
- 子结构修补（骨架精化、片段连接）
- 分子优化

**核心技术：**
- E(n)-等变图神经网络
- PyTorch Lightning 框架
- 去噪扩散概率模型（DDPM）
- RDKit 分子处理
- BioPython 蛋白质结构处理

**支持的数据集：**
- CrossDocked 基准测试
- Binding MOAD

---

## 核心架构

### 主入口点

#### 训练
- **文件：** `DiffSBDD/train.py`（123 行）
- **用途：** 使用 PyTorch Lightning 的主训练脚本
- **关键函数：**
  - `merge_args_and_yaml()` - 合并命令行参数与配置
  - `merge_configs()` - 合并配置以恢复训练
- **特性：**
  - YAML 配置加载
  - WandB 日志集成
  - 模型检查点
  - 多 GPU 支持（DDP）
  - 恢复训练能力

#### 推理脚本
1. **generate_ligands.py**（2255 字节）- 为口袋生成配体
2. **test.py**（7308 字节）- 测试集评估
3. **inpaint.py**（8952 字节）- 子结构修补
4. **optimize.py**（9554 字节）- 分子优化

---

## 模块详情

## 1. 核心训练模块

### 1.1 Lightning 模块
**文件：** `DiffSBDD/lightning_modules.py`（38178 字节，915 行）

**用途：** DDPM 训练和评估的 PyTorch Lightning 封装器

#### LigandPocketDDPM 类
- **继承关系：** `pl.LightningModule`
- **关键方法：**
  - `__init__()` - 初始化模型、优化器、指标
  - `forward()` - 通过扩散模型的前向传播
  - `training_step()` - 带损失计算的训练步骤
  - `validation_step()` - 验证步骤
  - `configure_optimizers()` - 设置优化器和调度器
  - `sample_batch()` - 在评估期间采样配体
- **特性：**
  - 三种 DDPM 模式：
    - `joint`: 配体和口袋的联合分布
    - `pocket_conditioning`: 条件 DDPM
    - `pocket_conditioning_simple`: 简化条件 DDPM
  - 数据增强（噪声、旋转）
  - 梯度裁剪
  - 评估指标计算
  - 可视化生成
  - WandB 日志记录

**支持的模式：**
- 联合训练
- 口袋条件化
- CA（C-alpha）表示
- 全原子表示

---

## 2. 等变扩散模块

**位置：** `DiffSBDD/equivariant_diffusion/`

**用途：** 具有 E(n)-等变性的核心扩散模型实现

### 2.1 扩散模型
**文件：** `equivariant_diffusion/en_diffusion.py`（49401 字节，1191 行）

#### EnVariationalDiffusion 类
- **用途：** E(n)-等变变分扩散模型
- **关键方法：**
  - `forward()` - 前向扩散过程
  - `sample()` - 从噪声中采样配体
  - `compute_loss()` - 计算 VLB 或 L2 损失
  - `sample_p_zs_given_zt()` - 反向扩散步骤
  - `phi()` - 噪声预测网络调用
- **关键组件：**
  - 噪声调度（学习或预定义）
  - 大小分布处理
  - 虚拟节点支持
  - 归一化方案

**噪声调度：**
- 学习的（使用 VLB 目标）
- 多项式
- 余弦

**损失类型：**
- VLB（变分下界）
- L2（均方误差）

#### 支持类
- **GammaNetwork:** 可学习的噪声调度
- **PredefinedNoiseSchedule:** 固定噪声调度
- **DistributionNodes:** 节点数量分布

---

### 2.2 条件模型
**文件：** `equivariant_diffusion/conditional_model.py`（31796 字节）

#### ConditionalDDPM
- **用途：** 带口袋信息的条件扩散
- **特性：**
  - 口袋条件生成
  - 上下文嵌入
  - 修补支持

#### SimpleConditionalDDPM
- **用途：** 简化条件模型
- **特性：**
  - 降低复杂度
  - 更快的推理

---

### 2.3 动力学模块
**文件：** `equivariant_diffusion/dynamics.py`（7756 字节）

#### EGNNDynamics 类
- **用途：** E(n)-等变动力学网络
- **关键方法：**
  - `forward()` - 预测噪声/更新
  - `_forward()` - 内部前向传播
- **特性：**
  - EGNN 主干
  - 时间嵌入
  - 上下文条件化
  - 辅助损失支持

**架构：**
- 多层 EGNN
- 节点和边特征
- 坐标更新
- 类型预测

---

### 2.4 EGNN 实现
**文件：** `equivariant_diffusion/egnn_new.py`（14842 字节）

#### EGNN 类
- **用途：** E(n) 等变图神经网络
- **组件：**
  - 边卷积层
  - 坐标更新
  - 注意力机制
  - 归一化层

#### E_GCL（等变图卷积层）
- **关键特性：**
  - 位置消息传递
  - 特征消息传递
  - 边条件更新
  - 坐标变换

**实现特性：**
- 旋转/平移等变性
- 排列不变性
- 注意力权重
- 边特征

---

## 3. 数据处理

### 3.1 数据集模块
**文件：** `DiffSBDD/dataset.py`（2693 字节，71 行）

#### ProcessedLigandPocketDataset 类
- **用途：** 配体-口袋对的 PyTorch 数据集
- **关键方法：**
  - `__init__()` - 加载 NPZ 数据
  - `__getitem__()` - 获取单个样本
  - `collate_fn()` - 批量整理
- **数据处理：**
  - NPZ 文件加载
  - 自动居中
  - 基于掩码的分割
  - 变换支持

**数据结构：**
- 配体坐标
- 口袋坐标
- 原子/残基类型
- 批处理掩码

---

### 3.2 数据预处理

#### process_crossdock.py
**文件：** `DiffSBDD/process_crossdock.py`（17122 字节）

**用途：** 处理 CrossDocked 基准数据集
- **功能：**
  - 数据集下载
  - 结构解析
  - 特征提取
  - 训练/验证/测试分割
- **选项：**
  - `--no_H`: 去除氢原子
  - CA vs 全原子表示

#### process_bindingmoad.py
**文件：** `DiffSBDD/process_bindingmoad.py`（25281 字节）

**用途：** 处理 Binding MOAD 数据集
- **功能：**
  - 结构解析
  - 质量过滤
  - 特征提取
  - 数据集分割
- **选项：**
  - `--ca_only`: C-alpha 表示

---

## 4. 常量和配置

### 4.1 常量
**文件：** `DiffSBDD/constants.py`（16196 字节，184 行）

**内容：**
- **计算相关：** 数据类型（FLOAT_TYPE, INT_TYPE）
- **键参数：**
  - 每种原子类型的允许键
  - 键长（单键、双键、三键）
  - 共价半径
- **骨架几何：**
  - N-CA 距离
  - CA-C 距离
  - 键角
- **数据集参数：**
  - 原子编码器/解码器
  - 氨基酸编码器/解码器
  - 适用于 CrossDocked 和 Binding MOAD

**支持的原子类型：**
- 常见：C, N, O, S, P, F, Cl, Br, I
- 扩展：B, Si, As, Hg, Bi, Al

**氨基酸：**
- 所有 20 种标准氨基酸

---

### 4.2 配置文件
**位置：** `DiffSBDD/configs/`

**文件：**
- `crossdock_ca_cond.yml`（1442 字节）- CrossDocked, CA, 条件
- `crossdock_ca_joint.yml`（1431 字节）- CrossDocked, CA, 联合
- `crossdock_fullatom_cond.yml`（1448 字节）- CrossDocked, 全原子, 条件
- `crossdock_fullatom_joint.yml`（1435 字节）- CrossDocked, 全原子, 联合

**配置参数：**
- 数据集设置
- 模型超参数
- 训练参数
- 评估设置
- WandB 配置

---

## 5. 分析模块

**位置：** `DiffSBDD/analysis/`

**用途：** 分子分析、指标和可视化

### 5.1 分子构建
**文件：** `analysis/molecule_builder.py`（7874 字节）

**关键函数：**
- `build_molecule()` - 从原子/坐标构建 RDKit 分子
- `process_molecule()` - 清理和处理分子
- `get_bond_order()` - 从距离推断键级
- `uff_relax()` - UFF 力场松弛
- `filter_rd_mol()` - 过滤无效分子

**特性：**
- 基于距离的键检测
- 化合价检查
- 环检测
- 3-3 环相交过滤
- 净化

---

### 5.2 指标
**文件：** `analysis/metrics.py`（8995 字节）

#### BasicMolecularMetrics 类
- **指标：**
  - 原子类型分布
  - 键类型分布
  - 电荷分布
  - 化合价分布
  - 分子稳定性

#### MoleculeProperties 类
- **指标：**
  - QED（药物相似性定量估计）
  - SA Score（合成可及性）
  - LogP（分配系数）
  - Lipinski 五规则
  - 多样性（Tanimoto 相似度）

#### CategoricalDistribution 类
- **用途：** 比较原子/键分布
- **指标：**
  - Wasserstein 距离
  - KL 散度

---

### 5.3 对接
**文件：**
- `analysis/docking.py`（7011 字节）- SMINA 对接接口
- `analysis/docking_py27.py`（886 字节）- Python 2.7 兼容性

**函数：**
- `smina_score()` - 计算 SMINA 对接得分
- 蛋白质-配体评分
- 结合亲和力估计

**特性：**
- SMINA 集成
- Vina 评分函数
- 柔性对接

---

### 5.4 可视化
**文件：** `analysis/visualization.py`（16203 字节）

**函数：**
- `save_xyz_file()` - 保存 XYZ 坐标文件
- `visualize()` - 可视化单个结构
- `visualize_chain()` - 可视化扩散链
- `plot_molecule()` - 2D/3D 分子图
- `plot_distribution()` - 分布图

**可视化类型：**
- 3D 分子结构
- 扩散轨迹
- 指标分布
- 属性比较

---

### 5.5 SA Score
**位置：** `analysis/SA_Score/`

**文件：**
- `sascorer.py` - 合成可及性评分
- `README.md` - 文档

**用途：** 计算分子的合成可及性得分

---

## 6. 工具模块

### 6.1 几何工具
**文件：** `DiffSBDD/geometry_utils.py`（4242 字节）

**函数：**
- `rigid_transform_3D()` - Kabsch 算法对齐
- `get_dihedral()` - 计算二面角
- `rotation_matrix()` - 生成旋转矩阵
- `apply_rotation()` - 对坐标应用旋转

**用途：** 几何变换和计算

---

### 6.2 通用工具
**文件：** `DiffSBDD/utils.py`（7171 字节）

**类：**
- `Queue` - 梯度归一化队列

**函数：**
- `get_edges()` - 计算图边
- `remove_mean()` - 居中坐标
- `assert_mean_zero()` - 验证居中
- 辅助工具

---

### 6.3 生成脚本

#### generate_ligands.py
**文件：** `DiffSBDD/generate_ligands.py`（2255 字节）

**用途：** 为给定口袋生成配体
- **输入：**
  - 检查点文件
  - PDB 文件
  - 口袋规范（残基列表或参考配体）
- **输出：**
  - 带生成分子的 SDF 文件
- **选项：**
  - 样本数量
  - 分子大小
  - 净化
  - 力场松弛

---

#### inpaint.py
**文件：** `DiffSBDD/inpaint.py`（8952 字节）

**用途：** 子结构修补和片段连接
- **特性：**
  - 固定子结构
  - 添加新原子
  - 片段连接
  - 骨架精化
- **选项：**
  - 固定原子规范
  - 中心模式（配体/口袋）
  - 新节点数量

---

#### optimize.py
**文件：** `DiffSBDD/optimize.py`（9554 字节）

**用途：** 进化分子优化
- **目标：**
  - QED 优化
  - SA 分数优化
  - 自定义目标
- **算法：**
  - 基于种群的进化
  - Top-k 选择
  - 加噪-去噪循环
- **参数：**
  - 种群大小
  - 进化步骤
  - 时间步
  - Top-k 选择

---

#### test.py
**文件：** `DiffSBDD/test.py`（7308 字节）

**用途：** 在测试集上评估
- **特性：**
  - 批处理
  - 多个口袋
  - 指标计算
  - 结果保存
- **选项：**
  - 固定节点数
  - 节点数偏差
  - 最小节点数

---

## 7. 支持文件

### 7.1 环境
**文件：** `DiffSBDD/environment.yaml`（505 字节）

**依赖项：**
- Python 3.10.4
- PyTorch 1.12.1
- PyTorch Lightning 1.7.4
- RDKit 2022.03.2
- BioPython 1.79
- WandB 0.13.1
- PyTorch Scatter 2.0.9
- OpenBabel 3.1.1
- imageio, scipy, seaborn

---

### 7.2 文档
**文件：**
- `README.md`（12911 字节，269 行）- 全面文档
- `LICENSE`（1098 字节）- MIT 许可证
- `analysis/SA_Score/README.md` - SA Score 文档

---

### 7.3 Colab 集成
**位置：** `DiffSBDD/colab/`

**用途：** Google Colab notebook 用于简单实验

---

## 8. 数据目录结构

### 目录布局
```
DiffSBDD/data/
  ├── crossdocked/
  │   ├── processed_noH/
  │   │   ├── train.npz
  │   │   ├── val.npz
  │   │   └── test.npz
  │   └── size_distribution.npy
  ├── bindingmoad/
  │   ├── processed_noH/
  │   │   ├── train.npz
  │   │   ├── val.npz
  │   │   └── test.npz
  │   └── size_distribution.npy
  └── checkpoints/
      ├── crossdocked_ca_cond.ckpt
      ├── crossdocked_ca_joint.ckpt
      ├── crossdocked_fullatom_cond.ckpt
      └── crossdocked_fullatom_joint.ckpt
```

---

## 统计摘要

### 项目整体统计

#### 文件数量
- **Python 文件：** 22
- **配置文件：** 5（4 YAML + 1 environment）
- **文档文件：** 2 个 README 文件
- **许可证：** 1 个文件

#### 代码统计（行数）
- `lightning_modules.py`: 38,178 字节（915 行）
- `equivariant_diffusion/en_diffusion.py`: 49,401 字节（1191 行）
- `equivariant_diffusion/conditional_model.py`: 31,796 字节
- `process_bindingmoad.py`: 25,281 字节
- `process_crossdock.py`: 17,122 字节
- `constants.py`: 16,196 字节（184 行）
- `analysis/visualization.py`: 16,203 字节
- `equivariant_diffusion/egnn_new.py`: 14,842 字节
- `README.md`: 12,911 字节（269 行）
- `optimize.py`: 9,554 字节
- `inpaint.py`: 8,952 字节
- `analysis/metrics.py`: 8,995 字节
- `analysis/molecule_builder.py`: 7,874 字节
- `equivariant_diffusion/dynamics.py`: 7,756 字节
- `test.py`: 7,308 字节
- `utils.py`: 7,171 字节
- `analysis/docking.py`: 7,011 字节
- `geometry_utils.py`: 4,242 字节
- `train.py`: 4,010 字节（123 行）
- `dataset.py`: 2,693 字节（71 行）
- `generate_ligands.py`: 2,255 字节

**总估计行数：** ~3,000+ 行 Python 代码

---

### 模块分类

#### 1. 核心训练（2 个文件）
- 训练脚本
- Lightning 模块封装器

#### 2. 等变扩散（4 个文件）
- 主扩散模型
- 条件模型
- 动力学网络
- EGNN 实现

#### 3. 数据处理（3 个文件）
- 数据集类
- CrossDocked 处理器
- Binding MOAD 处理器

#### 4. 分析工具（6 个文件）
- 分子构建器
- 指标计算
- 对接接口
- 可视化
- SA Score

#### 5. 推理脚本（4 个文件）
- 配体生成
- 修补
- 优化
- 测试

#### 6. 工具（4 个文件）
- 常量
- 几何工具
- 通用工具
- 配置文件

---

### 功能组件

#### 分子生成模式
1. **从头设计：**
   - 从头生成配体
   - 口袋条件生成
   - 大小控制

2. **子结构修补：**
   - 骨架精化
   - 片段连接
   - R-基团装饰
   - 固定原子约束

3. **分子优化：**
   - 进化算法
   - 属性优化（QED, SA）
   - 基于种群的搜索

#### 模型变体
1. **架构：**
   - 联合模型（配体 + 口袋联合分布）
   - 条件模型（口袋 → 配体）
   - 简化条件模型

2. **口袋表示：**
   - 仅 C-alpha（CA）
   - 全原子表示

3. **训练：**
   - VLB 损失
   - L2 损失
   - 辅助损失

---

### 关键特性摘要

#### 1. E(n)-等变性
- 旋转等变性
- 平移等变性
- 排列不变性

#### 2. 扩散过程
- 前向扩散（加噪）
- 反向扩散（去噪）
- 灵活的噪声调度
- 可变分子大小

#### 3. 条件化
- 蛋白质口袋条件化
- 基于参考配体的口袋定义
- 基于残基的口袋规范

#### 4. 分子质量
- 键级推断
- 化合价检查
- 环检测
- 净化

#### 5. 评估
- 基本分子指标
- QED, SA, LogP
- Lipinski 规则
- 多样性指标
- 对接得分

#### 6. 数据集
- CrossDocked 基准
- Binding MOAD
- 自动预处理
- 训练/验证/测试分割

---

### 训练特性

#### 数据增强
- 高斯噪声添加
- 随机旋转
- 质心居中

#### 优化
- Adam 优化器
- 学习率调度
- 梯度裁剪
- 多 GPU 支持（DDP）

#### 监控
- WandB 日志记录
- 验证指标
- 样本可视化
- 链可视化
- 模型检查点

---

### 推理能力

#### 采样控制
- 样本数量
- 分子大小（固定或采样）
- 去噪步骤
- 温度控制

#### 后处理
- 片段过滤
- 分子净化
- 力场松弛
- 对接分数计算

#### 批处理
- 测试集评估
- 多个口袋
- 并行处理

---

## 预训练模型

可在 [Zenodo](https://zenodo.org/record/8183747) 上获取：

**CrossDocked 模型：**
- CA 条件
- CA 联合
- 全原子条件
- 全原子联合

**Binding MOAD 模型：**
- CA 条件
- CA 联合
- 全原子条件
- 全原子联合

---

## 基准测试

### CrossDocked 基准
- **规模：** 大规模蛋白质-配体对
- **来源：** Pocket2Mol 数据集
- **处理：** 自动化流程

### Binding MOAD
- **规模：** 精选数据库
- **来源：** http://www.bindingmoad.org/
- **处理：** 质量过滤

### 采样分子
- 可在 [Zenodo](https://zenodo.org/record/8239058) 上获取

---

## 使用工作流程

### 1. 从头设计工作流程
```bash
# 下载检查点
wget -P checkpoints/ <checkpoint_url>

# 生成配体
python generate_ligands.py checkpoints/model.ckpt \
    --pdbfile protein.pdb \
    --ref_ligand A:330 \
    --n_samples 20 \
    --outfile results.sdf
```

### 2. 片段连接工作流程
```bash
python inpaint.py checkpoints/model.ckpt \
    --pdbfile protein.pdb \
    --ref_ligand ligand.sdf \
    --fix_atoms fragments.sdf \
    --center ligand \
    --add_n_nodes 10 \
    --outfile linked.sdf
```

### 3. 优化工作流程
```bash
python optimize.py \
    --checkpoint checkpoints/model.ckpt \
    --pdbfile protein.pdb \
    --ref_ligand ligand.sdf \
    --objective qed \
    --population_size 100 \
    --evolution_steps 10 \
    --outfile optimized.sdf
```

### 4. 训练工作流程
```bash
# 新训练
python train.py --config configs/config.yml

# 恢复训练
python train.py --config configs/config.yml \
    --resume checkpoints/last.ckpt
```

---

## 技术亮点

### 创新贡献
1. **E(n)-等变架构：**
   - 首个用于 SBDD 的等变扩散模型
   - 保持 3D 对称性
   - 改进的样本质量

2. **灵活的条件化：**
   - 多种口袋表示
   - 参考配体或残基列表
   - 联合 vs 条件训练

3. **子结构修补：**
   - 固定原子约束
   - 片段连接
   - 骨架精化

4. **全面评估：**
   - 多种指标
   - 对接集成
   - 质量评估

---

## 依赖项兼容性

| 软件              | 版本      | 用途 |
|-------------------|-----------|------|
| Python            | 3.10.4    | 核心 |
| PyTorch           | 1.12.1    | 深度学习 |
| PyTorch Lightning | 1.7.4     | 训练框架 |
| CUDA              | 10.2.89   | GPU 加速 |
| RDKit             | 2022.03.2 | 化学信息学 |
| BioPython         | 1.79      | 蛋白质结构 |
| WandB             | 0.13.1    | 实验跟踪 |
| PyTorch Scatter   | 2.0.9     | 图操作 |
| OpenBabel         | 3.1.1     | 分子转换 |
| imageio           | 2.21.2    | 可视化 |
| SciPy             | 1.7.3     | 科学计算 |

---

## 文档元数据

- **生成基于：** 全面的代码库分析
- **项目路径：** `DiffSBDD/`
- **相对于：** `DiffB110/`
- **分析的总模块数：** 6 个主要模块
- **扫描的总文件数：** 30+ 个文件
- **Python 文件：** 22
- **配置文件：** 5
- **文档文件：** 2

---

## 快速参考：文件路径索引

### 核心脚本
- 训练：`DiffSBDD/train.py`
- 生成：`DiffSBDD/generate_ligands.py`
- 修补：`DiffSBDD/inpaint.py`
- 优化：`DiffSBDD/optimize.py`
- 测试：`DiffSBDD/test.py`

### 主模块
- Lightning 模块：`DiffSBDD/lightning_modules.py`
- 扩散模型：`DiffSBDD/equivariant_diffusion/en_diffusion.py`
- 条件模型：`DiffSBDD/equivariant_diffusion/conditional_model.py`
- 动力学：`DiffSBDD/equivariant_diffusion/dynamics.py`
- EGNN：`DiffSBDD/equivariant_diffusion/egnn_new.py`

### 数据处理
- 数据集：`DiffSBDD/dataset.py`
- CrossDocked：`DiffSBDD/process_crossdock.py`
- Binding MOAD：`DiffSBDD/process_bindingmoad.py`

### 分析
- 分子构建器：`DiffSBDD/analysis/molecule_builder.py`
- 指标：`DiffSBDD/analysis/metrics.py`
- 对接：`DiffSBDD/analysis/docking.py`
- 可视化：`DiffSBDD/analysis/visualization.py`

### 工具
- 常量：`DiffSBDD/constants.py`
- 几何：`DiffSBDD/geometry_utils.py`
- 工具：`DiffSBDD/utils.py`

### 配置
- 配置：`DiffSBDD/configs/*.yml`
- 环境：`DiffSBDD/environment.yaml`

---

**文档结束**
