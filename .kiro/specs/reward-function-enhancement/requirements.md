# 奖励函数增强需求文档

## 概述

本文档描述 DiffSBDD-DDPO 项目中奖励函数系统的增强需求，重点解决强化学习中的"奖励作弊"(reward hacking) 问题，并引入基于真实药物分子高维特征相似性的奖励机制。

## 背景与动机

### 当前奖励函数的局限性

现有的奖励函数系统包含以下组件：

1. **对接奖励 (DockingReward)**: 基于 AutoDock Vina 计算结合亲和力
2. **药物相似性奖励 (DruglikenessReward)**: 基于 QED 和 SA Score
3. **组合奖励 (CompositeReward)**: 加权组合多个奖励

**存在的问题**:

强化学习优化过程中，模型可能会"作弊"以获得高分数，而不是生成真正有用的分子：

- **对接分数作弊**: 生成化学上不合理但对接分数很好的结构（如过度扭曲的构象）
- **QED/SA 作弊**: 生成满足简单规则但缺乏真实药物特征的分子
- **物理不合理**: 生成的分子可能违反物理化学规律（如不合理的键长、键角）
- **缺乏多样性**: 模型可能收敛到少数几个高分但无实际价值的分子

### 解决方案：真实药物相似性奖励

引入一个新的奖励组件，通过与已知药物分子的高维特征相似性来约束生成：

**核心思想**: 生成的分子应该在多个维度上与真实的、已批准的药物分子相似，而不仅仅是优化单一的打分函数。

**优势**:
- 防止奖励作弊：真实药物的特征分布是复杂的，难以通过简单的作弊策略满足
- 隐式约束：自动包含了药物化学家积累的经验和知识
- 多样性保证：参考药物库的多样性可以引导生成多样化的候选分子
- 可解释性：可以分析生成分子与哪些已知药物相似

## 需求规格

### Requirement 1: UniDock GPU 加速对接

**User Story**: As a researcher, I want to use UniDock with GPU acceleration instead of smina, so that molecular docking can be computed much faster during RL training.

#### Acceptance Criteria

1. WHEN the system performs docking THEN it SHALL use UniDock's GPU-accelerated vina instead of smina
2. WHEN UniDock is initialized THEN it SHALL detect and utilize available GPUs
3. WHEN docking is performed THEN the system SHALL support batch processing on GPU for maximum throughput
4. WHEN GPU is not available THEN the system SHALL fall back to CPU-based docking with a warning
5. WHEN docking parameters are configured THEN the system SHALL support all standard Vina parameters (center, size, exhaustiveness)
6. WHEN docking completes THEN the system SHALL return binding affinity scores compatible with the existing reward normalization

**技术要点**:
- UniDock 支持 CUDA 加速，可以显著提升对接速度（10-100x）
- 需要安装 UniDock 并配置 GPU 环境
- 保持与现有 DockingReward 接口兼容
- 支持批量提交多个分子到 GPU 进行并行对接

**性能目标**:
- 单分子对接时间: < 0.5 秒 (GPU) vs ~3-5 秒 (smina CPU)
- 批量对接 (128 molecules): < 30 秒 (GPU) vs ~640 秒 (smina serial)
- GPU 利用率: > 80%

**配置示例**:
```yaml
reward:
  docking:
    backend: "unidock"  # 新增: "unidock" 或 "smina"
    use_gpu: true
    gpu_id: 0
    batch_size: 32  # GPU 批处理大小
```

### Requirement 2: UniDock 与 Ray 并行集成

**User Story**: As a developer, I want to integrate UniDock GPU docking with Ray parallelization, so that multiple GPUs can be utilized efficiently for large batches.

#### Acceptance Criteria

1. WHEN multiple GPUs are available THEN the system SHALL distribute docking tasks across GPUs using Ray
2. WHEN Ray workers are initialized THEN each worker SHALL be assigned to a specific GPU
3. WHEN docking tasks are submitted THEN the system SHALL balance load across available GPUs
4. WHEN GPU memory is limited THEN the system SHALL adjust batch size automatically to prevent OOM errors
5. WHEN docking fails on GPU THEN the system SHALL retry on CPU with a warning

**实现策略**:
```python
@ray.remote(num_gpus=1)
def calculate_unidock_score(molecules, receptor, gpu_id):
    # 使用指定 GPU 进行批量对接
    return unidock.dock_batch(molecules, receptor, gpu_id=gpu_id)
```

**多 GPU 调度**:
- 自动检测可用 GPU 数量
- 将大批次分割到多个 GPU
- 支持异步提交和结果收集

### Requirement 3: 真实药物数据库

**User Story**: As a researcher, I want to use a curated database of approved drugs as reference, so that generated molecules can be compared against real drug-like molecules.

### Requirement 3: 真实药物数据库

**User Story**: As a researcher, I want to use a curated database of approved drugs as reference, so that generated molecules can be compared against real drug-like molecules.

#### Acceptance Criteria

1. WHEN the system initializes THEN it SHALL load a database of FDA-approved drugs or similar curated drug datasets
2. WHEN loading the database THEN the system SHALL extract and store molecular structures in RDKit Mol format
3. WHEN the database is loaded THEN the system SHALL compute and cache high-dimensional feature vectors for all reference drugs
4. IF the database file is not found THEN the system SHALL raise a FileNotFoundError with a clear message
5. WHEN the database is accessed THEN the system SHALL provide efficient lookup and similarity search capabilities

**数据源建议**:
- DrugBank (FDA-approved drugs)
- ChEMBL (bioactive molecules)
- ZINC (drug-like compounds)
- 自定义数据集（针对特定靶点的已知抑制剂）

### Requirement 4: 高维特征提取

**User Story**: As a developer, I want to extract multiple types of molecular features, so that similarity can be measured in a comprehensive way.

### Requirement 4: 高维特征提取

**User Story**: As a developer, I want to extract multiple types of molecular features, so that similarity can be measured in a comprehensive way.

#### Acceptance Criteria

1. WHEN a molecule is provided THEN the system SHALL extract the following feature types:
   - Molecular fingerprints (ECFP, MACCS keys, etc.)
   - Physicochemical properties (MW, LogP, TPSA, etc.)
   - Structural descriptors (ring count, rotatable bonds, etc.)
   - 3D shape descriptors (if applicable)
   - Pharmacophore features

2. WHEN extracting features THEN the system SHALL normalize each feature type to [0, 1] range or standardize to zero mean and unit variance

3. WHEN feature extraction fails for a molecule THEN the system SHALL return None and log a warning

4. WHEN features are computed THEN the system SHALL cache results to avoid redundant computation

**特征类型详细说明**:

| 特征类型 | 描述 | 维度 | 工具 |
|---------|------|------|------|
| ECFP4 | Extended Connectivity Fingerprint | 1024-2048 | RDKit |
| MACCS | MACCS keys | 166 | RDKit |
| Physicochemical | MW, LogP, TPSA, HBD, HBA, etc. | ~20 | RDKit |
| Structural | Ring count, aromatic rings, rotatable bonds | ~15 | RDKit |
| Shape (optional) | 3D shape descriptors | Variable | RDKit/OpenBabel |

### Requirement 5: 相似性度量

**User Story**: As a researcher, I want to measure similarity between generated and reference molecules using multiple metrics, so that I can capture different aspects of drug-likeness.

### Requirement 5: 相似性度量

**User Story**: As a researcher, I want to measure similarity between generated and reference molecules using multiple metrics, so that I can capture different aspects of drug-likeness.

#### Acceptance Criteria

1. WHEN comparing two molecules THEN the system SHALL support the following similarity metrics:
   - Tanimoto similarity (for fingerprints)
   - Euclidean distance (for continuous features)
   - Cosine similarity (for feature vectors)
   - Weighted combination of multiple metrics

2. WHEN computing similarity THEN the system SHALL find the k most similar reference drugs (default k=10)

3. WHEN multiple similarity metrics are used THEN the system SHALL combine them using configurable weights

4. WHEN similarity is computed THEN the system SHALL return a score in [0, 1] range where 1 means identical

**相似性计算公式**:

```
similarity_total = w_fp * sim_fingerprint + w_phys * sim_physicochemical + w_struct * sim_structural

其中:
- sim_fingerprint: Tanimoto similarity of ECFP4
- sim_phys: 1 - normalized_euclidean_distance(physicochemical features)
- sim_struct: 1 - normalized_euclidean_distance(structural features)
- w_fp, w_phys, w_struct: 可配置权重（默认 0.5, 0.3, 0.2）
```

### Requirement 6: DrugSimilarityReward 实现

**User Story**: As a researcher, I want a reward function that encourages similarity to real drugs, so that generated molecules are more likely to be drug-like.

### Requirement 6: DrugSimilarityReward 实现

**User Story**: As a researcher, I want a reward function that encourages similarity to real drugs, so that generated molecules are more likely to be drug-like.

#### Acceptance Criteria

1. WHEN DrugSimilarityReward is initialized THEN it SHALL load the reference drug database and precompute features

2. WHEN calculate_rewards() is called with a batch of molecules THEN the system SHALL:
   - Extract features for each generated molecule
   - Compute similarity to k nearest reference drugs
   - Aggregate similarities into a single reward score
   - Return rewards in [0, 1] range

3. WHEN a generated molecule is invalid THEN the system SHALL assign a penalty reward of 0.0

4. WHEN computing rewards THEN the system SHALL support the following aggregation methods:
   - Max similarity (most similar drug)
   - Mean of top-k similarities
   - Weighted mean with distance-based weights

5. WHEN the reward is computed THEN the system SHALL optionally log the most similar reference drugs for interpretability

**聚合方法**:

```python
# Method 1: Max similarity
reward = max(similarities_to_k_nearest)

# Method 2: Mean of top-k
reward = mean(similarities_to_k_nearest)

# Method 3: Weighted mean (closer drugs have higher weight)
weights = softmax(-distances_to_k_nearest / temperature)
reward = sum(weights * similarities_to_k_nearest)
```

### Requirement 7: 与现有奖励函数集成

**User Story**: As a developer, I want to integrate DrugSimilarityReward into the existing CompositeReward system, so that it can be combined with docking and druglikeness rewards.

### Requirement 7: 与现有奖励函数集成

**User Story**: As a developer, I want to integrate DrugSimilarityReward into the existing CompositeReward system, so that it can be combined with docking and druglikeness rewards.

#### Acceptance Criteria

1. WHEN DrugSimilarityReward is added to CompositeReward THEN it SHALL follow the same BaseReward interface

2. WHEN CompositeReward combines rewards THEN DrugSimilarityReward SHALL be weighted alongside other rewards

3. WHEN training configuration is updated THEN the system SHALL support configuring DrugSimilarityReward weight

4. WHEN DrugSimilarityReward fails THEN CompositeReward SHALL handle the exception gracefully and use fallback value

**配置示例**:

```yaml
reward:
  weights:
    docking: 0.4
    druglikeness: 0.2
    drug_similarity: 0.4  # 新增
  docking:
    backend: "unidock"  # 使用 UniDock GPU 加速
    use_gpu: true
    gpu_id: 0
  drug_similarity:
    database_path: "./data/drugbank_approved.sdf"
    k_nearest: 10
    aggregation: "weighted_mean"
    feature_weights:
      fingerprint: 0.5
      physicochemical: 0.3
      structural: 0.2
```

### Requirement 8: 性能优化

**User Story**: As a developer, I want the similarity computation to be efficient, so that it doesn't become a bottleneck in training.

### Requirement 8: 性能优化

**User Story**: As a developer, I want the similarity computation to be efficient, so that it doesn't become a bottleneck in training.

#### Acceptance Criteria

1. WHEN the reference database is large (>10,000 drugs) THEN the system SHALL use approximate nearest neighbor search (e.g., FAISS, Annoy)

2. WHEN features are computed THEN the system SHALL cache results to avoid redundant computation

3. WHEN computing similarities in parallel THEN the system SHALL support Ray parallelization similar to DockingReward

4. WHEN batch processing THEN the system SHALL vectorize operations where possible

5. WHEN using UniDock GPU docking THEN the system SHALL achieve at least 10x speedup compared to smina CPU

**性能目标**:
- 参考数据库加载: < 10 秒 (10,000 drugs)
- 单分子特征提取: < 0.1 秒
- 单分子相似性搜索: < 0.01 秒 (with ANN index)
- 批量处理 (256 molecules): < 5 秒
- UniDock GPU 对接 (单分子): < 0.5 秒
- UniDock GPU 对接 (128 batch): < 30 秒

### Requirement 9: 可解释性与调试

**User Story**: As a researcher, I want to understand why a molecule received a certain similarity score, so that I can interpret the results and debug issues.

### Requirement 9: 可解释性与调试

**User Story**: As a researcher, I want to understand why a molecule received a certain similarity score, so that I can interpret the results and debug issues.

#### Acceptance Criteria

1. WHEN a molecule is evaluated THEN the system SHALL optionally return the top-k most similar reference drugs with their names and similarity scores

2. WHEN logging is enabled THEN the system SHALL log:
   - Feature extraction statistics
   - Similarity distribution across the batch
   - Most frequently matched reference drugs
   - UniDock GPU utilization and timing statistics

3. WHEN visualization is requested THEN the system SHALL provide utilities to:
   - Plot similarity distributions
   - Visualize generated molecule alongside most similar reference drugs
   - Show feature space projections (t-SNE, UMAP)

4. WHEN debugging THEN the system SHALL support saving intermediate results (features, similarities) to disk

### Requirement 10: 防止模式崩溃

**User Story**: As a researcher, I want to prevent the model from collapsing to a few high-similarity molecules, so that diversity is maintained.

#### Acceptance Criteria

1. WHEN computing rewards THEN the system SHALL optionally apply diversity penalties:
   - Penalize molecules too similar to previously generated ones
   - Encourage exploration of different regions of chemical space

2. WHEN diversity penalty is enabled THEN the system SHALL maintain a buffer of recently generated molecules

3. WHEN a new molecule is too similar to buffered molecules THEN the system SHALL reduce its reward

4. WHEN the buffer is full THEN the system SHALL use a sliding window or reservoir sampling strategy

**多样性惩罚公式**:

```python
diversity_penalty = max(similarities_to_recent_molecules)
final_reward = base_reward * (1 - alpha * diversity_penalty)

其中 alpha 是可配置的惩罚强度 (default: 0.2)
```

## 技术设计要点

### UniDock 集成

**安装与配置**:
```bash
# 安装 UniDock
conda install -c conda-forge unidock

# 或从源码编译以支持特定 CUDA 版本
git clone https://github.com/dptech-corp/Uni-Dock.git
cd Uni-Dock
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # 根据 GPU 架构调整
make -j8
```

**API 使用**:
```python
from unidock import UniDock

# 初始化 UniDock
docker = UniDock(
    receptor="receptor.pdbqt",
    center=[x, y, z],
    size=[sx, sy, sz],
    gpu_id=0
)

# 批量对接
results = docker.dock_batch(
    ligands=["ligand1.pdbqt", "ligand2.pdbqt", ...],
    exhaustiveness=8
)

# 获取分数
scores = [r.affinity for r in results]
```

### 文件结构

### 文件结构

```
ddpo/rewards/
├── docking.py                # 更新: 支持 UniDock 后端
├── docking_backends/         # 新增: 对接后端抽象
│   ├── __init__.py
│   ├── base.py              # BaseDockingBackend 接口
│   ├── smina.py             # Smina 后端（保留兼容）
│   └── unidock.py           # UniDock GPU 后端
├── drug_similarity.py        # DrugSimilarityReward 实现
├── feature_extractor.py      # 特征提取工具
└── similarity_metrics.py     # 相似性度量函数

data/
└── reference_drugs/
    ├── drugbank_approved.sdf
    ├── chembl_bioactive.sdf
    └── features_cache/
        ├── drugbank_features.pkl
        └── ann_index.faiss
```

### 关键类设计

```python
# 对接后端抽象
class BaseDockingBackend:
    """对接后端基类"""
    def dock_single(ligand_file, receptor_file, **kwargs) -> float
    def dock_batch(ligand_files, receptor_file, **kwargs) -> List[float]

class UniDockBackend(BaseDockingBackend):
    """UniDock GPU 加速后端"""
    def __init__(self, gpu_id=0, batch_size=32)
    def dock_batch(ligand_files, receptor_file, **kwargs) -> List[float]

class SminaBackend(BaseDockingBackend):
    """Smina CPU 后端（兼容性）"""
    def dock_single(ligand_file, receptor_file, **kwargs) -> float

# 更新后的 DockingReward
class DockingReward(BaseReward):
    def __init__(self, backend="unidock", use_gpu=True, **kwargs)
    def calculate_rewards(data: DataProto) -> DataProto

# 特征提取与相似性
class FeatureExtractor:
    """提取分子的多维特征"""
    def extract_fingerprint(mol) -> np.ndarray
    def extract_physicochemical(mol) -> np.ndarray
    def extract_structural(mol) -> np.ndarray
    def extract_all(mol) -> Dict[str, np.ndarray]

class SimilarityMetrics:
    """计算分子间相似性"""
    def tanimoto_similarity(fp1, fp2) -> float
    def euclidean_similarity(vec1, vec2) -> float
    def cosine_similarity(vec1, vec2) -> float
    def combined_similarity(features1, features2, weights) -> float

class DrugDatabase:
    """管理参考药物数据库"""
    def __init__(database_path, cache_dir)
    def load_molecules() -> List[Mol]
    def build_feature_index()
    def search_similar(query_features, k) -> List[Tuple[int, float]]

class DrugSimilarityReward(BaseReward):
    """基于真实药物相似性的奖励函数"""
    def __init__(database_path, k_nearest, aggregation, feature_weights)
    def calculate_rewards(data: DataProto) -> DataProto
```

## 实验验证

### 验证指标

1. **UniDock 性能验证**:
   - 对接速度对比: UniDock GPU vs smina CPU
   - GPU 利用率监控
   - 批处理吞吐量测试
   - 多 GPU 扩展性测试

2. **奖励作弊检测**:
   - 生成分子的化学有效性 (应保持 > 80%)
   - 生成分子的物理合理性 (键长、键角分布)
   - 生成分子的多样性 (Tanimoto 相似度分布)

3. **药物相似性**:
   - 生成分子与参考药物的平均相似度
   - 生成分子覆盖的参考药物数量
   - 特征空间的分布对比 (t-SNE 可视化)

4. **训练稳定性**:
   - 奖励曲线的平滑性
   - 是否出现模式崩溃
   - 训练速度影响

### 对比实验

| 实验组 | 奖励函数配置 | 对接后端 | 预期结果 |
|--------|-------------|---------|---------|
| Baseline | Docking (0.7) + Druglikeness (0.3) | smina CPU | 可能出现奖励作弊，速度慢 |
| +UniDock | Docking (0.7) + Druglikeness (0.3) | UniDock GPU | 速度提升 10-100x |
| +Similarity | Docking (0.4) + Druglikeness (0.2) + Similarity (0.4) | UniDock GPU | 减少作弊，提高真实性 |
| +Diversity | 上述 + 多样性惩罚 | UniDock GPU | 进一步提高多样性 |

### UniDock 性能基准测试

**测试场景**:
- 单分子对接: 1 个分子 × 1000 次
- 小批量: 32 个分子 × 100 批次
- 大批量: 128 个分子 × 50 批次
- 多 GPU: 256 个分子 × 2/4/8 GPUs

**预期结果**:
```
Backend         | Single | Batch-32 | Batch-128 | Multi-GPU (4x)
----------------|--------|----------|-----------|----------------
smina CPU       | 3.5s   | 112s     | 448s      | N/A
UniDock 1 GPU   | 0.4s   | 8s       | 28s       | N/A
UniDock 4 GPUs  | N/A    | 2s       | 7s        | 1.8s (256 mols)
Speedup         | 8.75x  | 14x      | 16x       | ~250x (vs serial)
```

## 未来扩展

1. **靶点特异性数据库**: 使用针对 RE-CmeB 或类似靶点的已知抑制剂作为参考
2. **动态权重调整**: 训练过程中自动调整各奖励组件的权重
3. **对抗性训练**: 引入判别器网络识别"作弊"分子
4. **多目标帕累托优化**: 同时优化多个目标而不是简单加权
5. **UniDock 高级特性**: 
   - 柔性对接 (flexible docking)
   - 多构象采样
   - 结合自由能计算
6. **分布式 GPU 集群**: 跨节点多 GPU 并行对接

## 参考文献

1. **UniDock**: "Uni-Dock: GPU-Accelerated Docking Enables Ultralarge Virtual Screening" (Yu et al., 2023)
2. **Reward Hacking in RL**: "Specification gaming: the flip side of AI ingenuity" (DeepMind, 2020)
3. **Molecular Similarity**: "Molecular similarity in drug design" (Martin et al., 2002)
4. **Drug-likeness**: "Quantifying the chemical beauty of drugs" (Bickerton et al., 2012)
5. **RL for Molecule Generation**: "Molecular de novo design through deep reinforcement learning" (Olivecrona et al., 2017)

---

**文档状态**: 已更新 - 添加 UniDock GPU 加速需求  
**创建时间**: 2026-02-28  
**最后更新**: 2026-03-06
