# Design Document: DiffSBDD-RLPF Integration

## Overview

本设计文档描述如何将 DiffSBDD（基于口袋条件的 E(n)-等变扩散模型）与 RLPF 框架集成，实现 DDPO 训练以优化针对 RE-CmeB 蛋白的分子生成。

**核心设计原则**：RL 框架透明性 —— 口袋数据完全封装在 DiffSBDDModel 和 DockingReward 内部，DDPOTrainer、EDMRollout、EDMActor 代码保持不变。

## Architecture

### 组件复用策略

| 组件 | 来源 | 修改程度 | 说明 |
|------|------|----------|------|
| **DDPOTrainer** | RLPF | 按需修改 | 协调训练流程，可能需要小幅调整以适配新组件 |
| **main_edm.py** | RLPF | 按需修改 | 入口脚本，替换组件初始化（EDMModel→DiffSBDDModel, ForceReward→自定义奖励） |
| **EDMActor** | RLPF | **直接复用** | PPO 策略更新，完全不修改 |
| **EDMRollout** | RLPF | 按需修改 | 轨迹采样，可能需要适配 DiffSBDDModel |
| **Filter** | RLPF | 替换 | 使用 DiffSBDDFilter（复用 DiffSBDD 的 build_molecule） |
| **ForceReward** | RLPF | **不使用** | 替换为用户自定义奖励（如 DockingReward） |
| **EDMModel** | RLPF | **不使用** | 替换为封装后的 DiffSBDDModel |
| **DiffSBDD** | DiffSBDD | 只读引用 | 扩散模型核心，不修改，仅作为依赖引用 |

### 关键设计决策

**1. 复用 EDMActor 的影响**

由于直接复用 RLPF 的 EDMActor，DiffSBDDModel 必须实现 EDMActor 调用的接口：

```python
# EDMActor.calculate_loss() 调用的方法
node_mask, edge_mask = self.model.get_mask(nodesxsample, batch_size, max_n_nodes)
_, log_prob, _, _ = self.model.sample_p_zs_given_zt(s, t, latents, node_mask, edge_mask, prev_sample=...)
```

这意味着 DiffSBDDModel 需要：
- 实现 `get_mask()` 返回 RLPF 格式的 mask `[B, N, 1]`
- 实现 `sample_p_zs_given_zt()` 接受 RLPF 格式输入，返回 `(zs, log_prob, mu, sigma)`
- 内部进行 RLPF ↔ DiffSBDD 格式转换

**2. 数据格式转换**

由于 EDMActor 期望 RLPF 格式 `[B, N, ...]`，而 DiffSBDD 内部使用 flat 格式 `[total_nodes, ...]`：
- DiffSBDDModel 在接口处进行格式转换
- 内部调用 DiffSBDD 时使用 flat 格式
- 返回给 RLPF 时转换为 padded 格式

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    DDPOTrainer (RLPF, 按需修改)                    │   │
│  │                                                                   │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │   │
│  │   │ EDMRollout  │───▶│ Rewarder    │───▶│ EDMActor    │          │   │
│  │   │ (按需修改)   │    │             │    │ (直接复用)   │          │   │
│  │   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │   │
│  │          │                  │                  │                  │   │
│  └──────────┼──────────────────┼──────────────────┼─────────────────┘   │
│             │                  │                  │                      │
│             ▼                  ▼                  ▼                      │
│  ┌─────────────────────┐  ┌─────────────────────┐                       │
│  │   DiffSBDDModel     │  │   CompositeReward   │                       │
│  │   (新建)             │  │   (新建)             │                       │
│  │                     │  │                     │                       │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │                       │
│  │  │ Fixed Pocket  │  │  │  │ DockingReward │  │                       │
│  │  │ (RE-CmeB)     │  │  │  │ (Vina + Ray)  │  │                       │
│  │  └───────────────┘  │  │  ├───────────────┤  │                       │
│  │                     │  │  │ DruglikeReward│  │                       │
│  │  ┌───────────────┐  │  │  │ (QED + SA)    │  │                       │
│  │  │ EnVariational │  │  │  └───────────────┘  │                       │
│  │  │ Diffusion     │  │  │                     │                       │
│  │  └───────────────┘  │  └─────────────────────┘                       │
│  │                     │                                                │
│  │  格式转换层:         │                                                │
│  │  RLPF [B,N,D] ↔     │                                                │
│  │  DiffSBDD [total,D] │                                                │
│  └─────────────────────┘                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DiffSBDD (只读引用，不修改)                            │
│                                                                          │
│  EnVariationalDiffusion.sample_p_zs_given_zt(zt_lig, zt_pocket, ...)   │
│  EGNNDynamics.forward(xh_atoms, xh_residues, t, masks)                  │
│  build_molecule(), process_molecule()                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. DiffSBDDModel (新建)

**职责**：封装 DiffSBDD 模型，提供与 EDMModel 相同的接口，内部处理口袋条件。

**位置**：`ddpo/integration/diffsbdd_model.py`

```python
class DiffSBDDModel(BaseModel):
    """
    Adapter wrapping DiffSBDD's EnVariationalDiffusion for RLPF framework.
    Pocket data is loaded once at initialization and reused for all sampling.
    """
    
    def __init__(self, checkpoint_path: str, pocket_pdb_path: str, device: torch.device):
        """
        Args:
            checkpoint_path: Path to pretrained DiffSBDD checkpoint
            pocket_pdb_path: Path to RE-CmeB PDB file
            device: torch device (cuda/cpu)
        """
        self.model: EnVariationalDiffusion  # DiffSBDD core model
        self.pocket: Dict[str, torch.Tensor]  # Fixed pocket data
        self.device: torch.device
    
    def sample(self, n_samples: int, n_nodes: int, node_mask: torch.Tensor, 
               edge_mask: torch.Tensor, context: Optional[torch.Tensor] = None,
               timestep: int = 1000) -> DataProto:
        """
        Generate molecules conditioned on fixed pocket.
        
        Returns:
            DataProto containing:
            - 'x': coordinates [B, N, 3]
            - 'h': atom types [B, N, F]
            - 'latents': list of latent states
            - 'log_probs': [B, T] log probabilities
            - 'timesteps': list of timestep values
            - 'mus': list of mean tensors
            - 'sigmas': list of sigma tensors
        """
    
    def sample_p_zs_given_zt(self, s: int, t: int, zt: torch.Tensor,
                             node_mask: torch.Tensor, edge_mask: torch.Tensor,
                             context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single denoising step with log probability computation.
        
        Returns:
            (zs, log_prob, mu, sigma) - only ligand-related tensors
        """
    
    def compute_log_p_zs_given_zt(self, zs: torch.Tensor, mu: torch.Tensor, 
                                   sigma: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Gaussian log probability.
        
        Formula: log_prob = -0.5 * sum((zs - mu)^2 / sigma^2) - sum(log(sigma)) - 0.5 * D * log(2π)
        """
    
    def load(self, model_path: str) -> None:
        """Load pretrained weights."""
    
    def to(self, device: torch.device) -> 'DiffSBDDModel':
        """Move model to device."""
```

### 2. 口袋数据加载 (复用 DiffSBDD)

**职责**：加载和预处理蛋白口袋数据。

**实现方式**：直接复用 DiffSBDD 的 `LigandPocketDDPM.prepare_pocket()` 方法，无需单独实现。

**DiffSBDD 已有实现** (`DiffSBDD/lightning_modules.py`):
```python
# LigandPocketDDPM.prepare_pocket() 已实现:
# - PDB 解析 (BioPython PDBParser)
# - CA 原子坐标提取
# - 残基类型 one-hot 编码
# - 坐标居中 (在 generate_ligands 中通过 scatter_mean 实现)

def prepare_pocket(self, biopython_residues, repeats=1):
    # 提取 CA 原子坐标
    pocket_coord = torch.tensor(np.array(
        [res['CA'].get_coord() for res in biopython_residues]), ...)
    # 残基类型编码
    pocket_types = torch.tensor(
        [self.pocket_type_encoder[three_to_one(res.get_resname())]
         for res in biopython_residues], ...)
    # one-hot 编码
    pocket_one_hot = F.one_hot(pocket_types, num_classes=len(self.pocket_type_encoder))
    # 返回口袋数据字典
    return {'x': ..., 'one_hot': ..., 'size': ..., 'mask': ...}
```

**在 DiffSBDDModel 中的使用**:
```python
class DiffSBDDModel:
    def __init__(self, checkpoint_path, pocket_pdb_path, ref_ligand_path, device):
        # 加载 LigandPocketDDPM 模型
        self.model = LigandPocketDDPM.load_from_checkpoint(checkpoint_path)
        
        # 使用 DiffSBDD 的方式加载口袋
        pdb_struct = PDBParser(QUIET=True).get_structure('', pocket_pdb_path)[0]
        residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand_path)
        self.pocket = self.model.prepare_pocket(residues, repeats=1)
```

### 3. MaskConverter (新建)

**职责**：在 DiffSBDD 和 RLPF 的 mask 格式之间转换。

**位置**：`ddpo/integration/mask_utils.py`

```python
def batch_mask_to_node_mask(batch_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert DiffSBDD batch_mask to RLPF node_mask.
    
    Args:
        batch_mask: [total_nodes] tensor with batch indices
    
    Returns:
        node_mask: [B, N, 1] tensor with 0/1 values
    
    Raises:
        ValueError: If batch_mask contains invalid values
    """

def node_mask_to_batch_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert RLPF node_mask to DiffSBDD batch_mask.
    
    Args:
        node_mask: [B, N, 1] tensor with 0/1 values
    
    Returns:
        batch_mask: [total_valid_nodes] tensor with batch indices
    
    Raises:
        ValueError: If node_mask contains invalid values
    """
```

### 4. DockingReward (新建)

**职责**：使用 AutoDock Vina 计算对接分数，支持 Ray 并行。

**位置**：`ddpo/rewards/docking.py`

```python
class DockingReward(BaseReward):
    """
    Compute docking scores using AutoDock Vina with Ray parallelization.
    Receptor data is loaded once at initialization.
    """
    
    def __init__(self, receptor_pdbqt_path: str, center: Tuple[float, float, float],
                 box_size: Tuple[float, float, float], num_workers: int = 8):
        """
        Args:
            receptor_pdbqt_path: Path to receptor PDBQT file
            center: (x, y, z) center of docking box
            box_size: (sx, sy, sz) size of docking box
            num_workers: Number of Ray workers for parallel docking
        """
        self.receptor_path: str
        self.center: Tuple[float, float, float]
        self.box_size: Tuple[float, float, float]
        self.num_workers: int
    
    def calculate_rewards(self, data: DataProto) -> DataProto:
        """
        Compute docking scores for a batch of molecules.
        
        Args:
            data: DataProto with 'x' coordinates and 'h' atom types
        
        Returns:
            DataProto with 'docking_scores' tensor (normalized by -10.0)
        """
```

### 5. DruglikenessReward (新建)

**职责**：计算 QED 和 SA 分数。

**位置**：`ddpo/rewards/druglikeness.py`

```python
class DruglikenessReward(BaseReward):
    """Compute QED and SA scores for drug-likeness evaluation."""
    
    def calculate_rewards(self, data: DataProto) -> DataProto:
        """
        Compute QED and SA scores.
        
        Returns:
            DataProto with:
            - 'qed_scores': [B] tensor in [0, 1]
            - 'sa_scores': [B] tensor normalized to [0, 1]
        """
```

### 6. CompositeReward (新建)

**职责**：组合多个奖励函数。

**位置**：`ddpo/rewards/composite.py`

```python
class CompositeReward(BaseReward):
    """Combine multiple reward functions with configurable weights."""
    
    def __init__(self, rewards: Dict[str, Tuple[BaseReward, float]]):
        """
        Args:
            rewards: Dict mapping name to (reward_function, weight) tuple
        """
    
    def calculate_rewards(self, data: DataProto) -> DataProto:
        """Compute weighted sum of all component rewards."""
    
    def update_weights(self, weights: Dict[str, float]) -> None:
        """Update reward weights dynamically."""
```

### 7. MoleculeConverter (新建)

**职责**：分子格式转换工具。

**位置**：`ddpo/utils/molecule_converter.py`

```python
def coords_to_rdkit_mol(x: torch.Tensor, h: torch.Tensor, 
                        dataset_info: Dict) -> Optional[Chem.Mol]:
    """Convert coordinates and atom types to RDKit Mol object."""

def rdkit_mol_to_pdbqt(mol: Chem.Mol) -> Optional[str]:
    """Convert RDKit Mol to PDBQT string for Vina docking."""

def process_molecule(mol: Chem.Mol) -> Chem.Mol:
    """Add hydrogens and optimize geometry."""
```

## Data Models

### DataProto (RLPF 现有，不修改)

```python
@dataclass
class DataProto:
    """Standard data transfer protocol in RLPF."""
    batch: TensorDict  # Contains tensors like 'x', 'h', 'log_probs'
    non_tensor_batch: Dict[str, Any]  # Non-tensor metadata
```

### Pocket Data Structure

```python
PocketData = TypedDict('PocketData', {
    'x': torch.Tensor,        # [N_residues, 3] centered coordinates
    'one_hot': torch.Tensor,  # [N_residues, 23] residue type encoding
    'mask': torch.Tensor,     # [N_residues] batch_mask format
    'size': int               # Number of residues
})
```

### Trajectory Data (sample() 返回)

```python
TrajectoryData = TypedDict('TrajectoryData', {
    'x': torch.Tensor,           # [B, N, 3] final coordinates
    'h': torch.Tensor,           # [B, N, F] final atom types
    'latents': List[torch.Tensor],  # T tensors of shape [B, N, D]
    'log_probs': torch.Tensor,   # [B, T] log probabilities
    'timesteps': List[int],      # T timestep values
    'mus': List[torch.Tensor],   # T mean tensors
    'sigmas': List[torch.Tensor] # T sigma tensors
})
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the prework analysis, the following correctness properties have been identified:

### Property 1: Mask Conversion Round-Trip

*For any* valid batch_mask tensor, converting to node_mask and back to batch_mask SHALL produce an output that matches the original batch_mask for all valid nodes.

**Validates: Requirements 2.3**

### Property 2: Pocket Coordinates Centering

*For any* loaded pocket data, the centroid of the coordinates SHALL be at the origin (within floating-point tolerance of 1e-6).

**Validates: Requirements 3.3**

### Property 3: Pocket Data Structure Completeness

*For any* successfully loaded pocket, the returned dictionary SHALL contain all required keys ('x', 'one_hot', 'mask', 'size') with correct tensor shapes.

**Validates: Requirements 3.4**

### Property 4: Docking Reward Normalization

*For any* computed docking score `s`, the normalized reward SHALL equal `s / -10.0`, producing positive values for negative (good) docking scores.

**Validates: Requirements 4.6**

### Property 5: QED Score Range

*For any* valid molecule, the computed QED score SHALL be in the range [0, 1].

**Validates: Requirements 5.2**

### Property 6: SA Score Normalization

*For any* computed SA score `sa` in range [1, 10], the normalized score SHALL equal `(10 - sa) / 9`, producing values in [0, 1].

**Validates: Requirements 5.4**

### Property 7: Composite Reward Weighted Sum

*For any* set of component rewards `r_i` with weights `w_i`, the total reward SHALL equal `sum(w_i * r_i)`.

**Validates: Requirements 6.2**

### Property 8: Validity Ratio Computation

*For any* batch of molecules with `num_valid` valid molecules out of `total`, the validity_ratio SHALL equal `num_valid / total`.

**Validates: Requirements 9.3**

### Property 9: Reward Rotation Invariance

*For any* molecule with coordinates `x` and any valid 3D rotation matrix `R`, the reward function SHALL produce `reward(R @ x) == reward(x)` (within floating-point tolerance).

**Validates: Requirements 10.3**

### Property 10: Log Probability Gaussian Formula

*For any* sample `zs`, mean `mu`, and standard deviation `sigma`, the computed log probability SHALL match the analytical Gaussian formula: `log_prob = -0.5 * sum((zs - mu)^2 / sigma^2) - sum(log(sigma)) - 0.5 * D * log(2π)`.

**Validates: Requirements 1.4, 11.1, 11.2**

### Property 11: Trajectory Log Probs Shape

*For any* sampling run with batch size `B` and `T` timesteps, the returned log_probs tensor SHALL have shape `[B, T]`.

**Validates: Requirements 11.4**

### Property 12: Sample Output Structure

*For any* call to `sample(n_samples, ...)`, the returned DataProto SHALL contain all required keys ('x', 'h', 'latents', 'log_probs', 'timesteps', 'mus', 'sigmas') with correct shapes relative to n_samples.

**Validates: Requirements 1.2**

### Property 13: PDBQT Conversion Consistency

*For any* valid RDKit Mol object, the PDBQT conversion SHALL produce a non-empty string containing valid PDBQT format markers.

**Validates: Requirements 12.3**

### Property 14: h Format Tensor Structure

*For any* call to DiffSBDDModel.sample(), the returned h SHALL be a tensor of shape [B, N, num_atom_types] containing one-hot encoded atom types.

**Note**: 不需要返回字典格式。RLPF Rollout 只存储 `h["categorical"]` 到 `batch["categorical"]`，下游模块直接读取 tensor。

**Validates: Requirements 14.1**

### Property 15: DiffSBDD build_molecule Compatibility

*For any* generated molecule with coordinates x and atom types h, the DiffSBDDFilter SHALL successfully call DiffSBDD's build_molecule function with crossdock dataset_info format.

**Validates: Requirements 14.2**

## Critical Interface Differences (新发现)

### 1. h 返回格式处理 🟢 RESOLVED

**问题描述**:
- RLPF EDMModel 返回: `h = {'integer': h_int, 'categorical': h_cat}` (dict)
- DiffSBDD 返回: `h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)` (直接 tensor)

**关键发现**: 
- RLPF Rollout 存储时只取 `h["categorical"]`，字典结构被拆开
- 下游模块 (Filter, Reward, Actor) 都直接读 `batch["categorical"]`，期望的是 tensor
- `h["integer"]` (电荷) 从 Rollout 开始就被丢弃，后续没有模块使用

**代码位置**:
```python
# RLPF EDMRollout.generate_minibatch() 存储方式
batch_data = {
    "categorical": h["categorical"],  # 只存 categorical，丢弃 integer
    ...
}

# RLPF Filter/Reward 使用方式
one_hot = samples.batch["categorical"]  # 直接读 tensor
atom_type = one_hot.argmax(dim=-1)
```

**解决方案**: 不需要转换成字典格式，直接存储 tensor
```python
# DiffSBDDModel.sample() 返回
h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)  # tensor [B, N, 10]

# DiffSBDDRollout.generate_minibatch() 存储
batch_data = {
    "categorical": h_lig,  # 直接存 tensor，不需要字典包装
    ...
}
```

**注意**: 虽然 RLPF EDMModel 内部返回字典格式，但这只是内部实现细节。
DiffSBDDModel 可以直接返回 tensor，只要 Rollout 正确存储到 `batch["categorical"]` 即可。

### 2. sample_p_zs_given_zt 返回值不匹配 🔴 HIGH

**问题描述**:
- RLPF EDMModel 返回: `(zs, log_prob, mu, sigma)` - 4 个值
- DiffSBDD ConditionalDDPM 返回: `(zs_lig, xh_pocket)` - 只有 2 个值，无 log_prob

**代码位置**:
```python
# DiffSBDD ConditionalDDPM.sample_p_zs_given_zt()
def sample_p_zs_given_zt(self, s, t, zt_lig, xh0_pocket, ligand_mask, pocket_mask, fix_noise=False):
    # ... 计算 mu, sigma 但不返回
    return zs_lig, xh0_pocket  # 只返回采样结果

# RLPF EDMModel.sample_p_zs_given_zt()
def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context=None, fix_noise=False, prev_sample=None):
    # ...
    return zs, log_p, mu, sigma  # 返回 4 个值
```

**解决方案**: 在 DiffSBDDModel 中重新实现，手动计算 mu, sigma, log_prob
```python
def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context=None, fix_noise=False, prev_sample=None):
    # 1. 转换格式
    zt_lig, lig_mask = self._rlpf_to_diffsbdd_format(zt, node_mask)
    pocket = self._replicate_pocket(batch_size)
    
    # 2. 计算 gamma 和 sigma/alpha
    gamma_s = self.ddpm.gamma(s)
    gamma_t = self.ddpm.gamma(t)
    sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
        self.ddpm.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)
    sigma_s = self.ddpm.sigma(gamma_s, target_tensor=zt_lig)
    sigma_t = self.ddpm.sigma(gamma_t, target_tensor=zt_lig)
    
    # 3. 调用 dynamics 获取 eps_t
    eps_t_lig, _ = self.ddpm.dynamics(zt_lig, xh_pocket, t, lig_mask, pocket['mask'])
    
    # 4. 计算 mu 和 sigma
    mu_lig = zt_lig / alpha_t_given_s[lig_mask] - \
             (sigma2_t_given_s / alpha_t_given_s / sigma_t)[lig_mask] * eps_t_lig
    sigma = sigma_t_given_s * sigma_s / sigma_t
    
    # 5. 采样 zs
    zs_lig, xh_pocket = self.ddpm.sample_normal_zero_com(mu_lig, xh_pocket, sigma, lig_mask, pocket['mask'])
    
    # 6. 计算 log_prob
    zs_rlpf = self._diffsbdd_to_rlpf_format(zs_lig, lig_mask, max_nodes)
    mu_rlpf = self._diffsbdd_to_rlpf_format(mu_lig, lig_mask, max_nodes)
    log_prob = self.compute_log_p_zs_given_zt(zs_rlpf, mu_rlpf, sigma, node_mask)
    
    return zs_rlpf, log_prob, mu_rlpf, sigma
```

### 3. 坐标居中方式不同 🟢 RESOLVED (自动处理)

**问题描述**:
- DiffSBDD: `remove_mean_batch(x_lig, x_pocket, lig_mask, pocket_mask)` - 同时处理配体和口袋，以配体质心为准
- RLPF: `_remove_mean_with_mask(x, node_mask)` - 只处理单一输入

**坐标居中的作用**：
- 保持 SE(3) 等变性（平移等变）
- 让分子质心在原点，避免数值漂移

**DiffSBDD 的居中逻辑**：
```python
def remove_mean_batch(cls, x_lig, x_pocket, lig_indices, pocket_indices):
    # 只用配体的质心（不是配体+口袋的联合质心）
    mean = scatter_mean(x_lig, lig_indices, dim=0)  # 配体质心
    x_lig = x_lig - mean[lig_indices]      # 配体居中
    x_pocket = x_pocket - mean[pocket_indices]  # 口袋也移动同样距离
    return x_lig, x_pocket
```

**解决方案**: 不需要额外处理！

DiffSBDD 的 `sample_normal_zero_com()` 内部已经调用了 `remove_mean_batch()`，会自动完成居中。
在 `DiffSBDDModel` 中只需要：
1. 调用 DiffSBDD 的方法（内部自动居中）
2. 只返回配体部分给 RLPF（口袋对 RLPF 不可见）

```python
# DiffSBDDModel.sample_p_zs_given_zt() 中
def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, ...):
    # 调用 DiffSBDD（内部会自动调用 remove_mean_batch）
    zs_lig, xh_pocket = self.ddpm.sample_normal_zero_com(...)
    # ↑ 这里面已经做了居中处理
    
    # 只返回配体部分给 RLPF
    zs = self._diffsbdd_to_rlpf_format(zs_lig, ...)  # 只有配体，已居中
    return zs, log_prob, mu, sigma
```

**工作量**: 无，DiffSBDD 内部已自动处理

### 4. build_molecule 函数不兼容 🟢 RESOLVED (复用 DiffSBDD)

**问题描述**:
- RLPF Filter 使用: `Model.EDM.qm9.rdkit_functions.build_molecule` (基于距离推断键，只支持 5 种原子)
- DiffSBDD 使用: `DiffSBDD.analysis.molecule_builder.build_molecule` (使用 OpenBabel，支持 10 种原子)

**解决方案**: 直接复用 DiffSBDD 的 `build_molecule`，不需要重新实现

创建 `DiffSBDDFilter` 类，复制 RLPF Filter 的逻辑，只修改 import：
```python
# ddpo/utils/diffsbdd_filter.py
# 只需要修改这一行 import，其他逻辑复制 RLPF Filter
from DiffSBDD.analysis.molecule_builder import build_molecule  # ← 改用 DiffSBDD 版本
from DiffSBDD.analysis.metrics import rdmol_to_smiles as mol2smiles
from DiffSBDD.constants import dataset_params

class DiffSBDDFilter:
    def __init__(self, ...):
        self.dataset_info = dataset_params['crossdock']  # 10 种原子
        # ... 其他逻辑与 RLPF Filter 相同
    
    def filter(self, data: DataProto) -> tuple:
        # 逻辑与 RLPF Filter 完全相同，只是 build_molecule 来自 DiffSBDD
        mol = build_molecule(pos, atom_type, self.dataset_info)
        smiles = mol2smiles(mol)
        # ...
```

**工作量**: 很小，只需复制 RLPF Filter 代码并修改 import 语句

### 5. dataset_info 结构不同 🟡 MEDIUM

**问题描述**:
- RLPF 使用 QM9 格式: `{'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}` (5 种原子)
- DiffSBDD 使用 crossdock 格式: `{'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}` (10 种原子)

**代码位置**:
```python
# DiffSBDD/constants.py
dataset_params['crossdock'] = {
    'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9},
    'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'],
    'aa_encoder': {...},  # 20 种氨基酸
    'aa_decoder': [...],
}
```

**解决方案**: 直接使用 DiffSBDD 的 dataset_info（不创建单独文件）
```python
# 在需要的地方直接导入
from DiffSBDD.constants import dataset_params
dataset_info = dataset_params['crossdock']
```

### 6. log_prob 格式要求 🟢 CLARIFIED

**关键发现**: RLPF 需要的是**每个时间步单独的 log_prob**，不是总 log_prob。

**数据流**:
```python
# 1. Model.sample() 返回每步的 log_prob 列表
logps = []  # List[Tensor[B]]
for s in reversed(range(0, self.T)):
    z, logp, mu, sigma = self.sample_p_zs_given_zt(...)
    logps.append(logp)  # 每步一个 [B] tensor

# 2. Rollout 堆叠成 [B, T+1]
batch_data = {
    "logps": torch.stack(logps, dim=1),  # [B, T+1]
    ...
}

# 3. Actor 逐步使用
for j in range(self.T):
    loss = self.calculate_loss(
        sample["logps"][:, j],  # 第 j 步的 log_prob [B]
        ...
    )
```

**数据形状**:

| 阶段 | 数据 | 形状 | 说明 |
|------|------|------|------|
| sample() 返回 | `logps` (列表) | `List[Tensor[B]]` | T+1 个 tensor |
| Rollout 存储 | `batch["logps"]` | `[B, T+1]` | 堆叠后 |
| Actor 使用 | `sample["logps"][:, j]` | `[B]` | 第 j 步 |

**为什么需要分开存储**:
- DDPO 算法对每个时间步单独计算 PPO 损失
- 每步的重要性采样比率: `ratio_j = exp(log_prob_new_j - log_prob_old_j)`
- 不是对整条轨迹求和，而是逐步更新

**DiffSBDDModel 实现要点**:
```python
def sample(self, n_samples, ...):
    logps = []
    for s in reversed(range(0, self.T)):
        z, logp, mu, sigma = self.sample_p_zs_given_zt(...)
        logps.append(logp)  # logp 形状: [B]
    
    # 返回列表，由 Rollout 堆叠
    return x, h, latents, logps, timesteps, mus, sigmas

def sample_p_zs_given_zt(self, s, t, zt, node_mask, ...):
    # ... 计算 mu, sigma
    # 计算单步 log_prob
    log_prob = self.compute_log_p_zs_given_zt(zs, mu, sigma, node_mask)
    return zs, log_prob, mu, sigma  # log_prob 形状: [B]
```

## Error Handling

### DiffSBDDModel Errors

| Error Condition | Exception Type | Message Format |
|----------------|----------------|----------------|
| Checkpoint file not found | `FileNotFoundError` | "Checkpoint not found: {path}" |
| Invalid checkpoint format | `ValueError` | "Invalid checkpoint format: {details}" |
| Pocket PDB not found | `FileNotFoundError` | "Pocket PDB not found: {path}" |
| Device mismatch | `RuntimeError` | "Model on {model_device}, expected {target_device}" |

### PocketLoader Errors

| Error Condition | Exception Type | Message Format |
|----------------|----------------|----------------|
| PDB file not found | `FileNotFoundError` | "PDB file not found: {path}" |
| PDB parsing failed | `ValueError` | "Failed to parse PDB: {error_details}" |
| No CA atoms found | `ValueError` | "No CA atoms found in PDB: {path}" |

### MaskConverter Errors

| Error Condition | Exception Type | Message Format |
|----------------|----------------|----------------|
| Invalid batch_mask values | `ValueError` | "Invalid batch_mask value {value}, expected range [0, {max}]" |
| Invalid node_mask values | `ValueError` | "Invalid node_mask values, expected 0 or 1" |

### DockingReward Errors

| Error Condition | Handling | Log Message |
|----------------|----------|-------------|
| PDBQT conversion failed | Return 0.0 reward | "WARNING: PDBQT conversion failed for molecule {idx}" |
| Vina calculation failed | Return 0.0 reward | "WARNING: Vina docking failed for molecule {idx}: {error}" |
| Receptor file not found | Raise exception | `FileNotFoundError("Receptor not found: {path}")` |

### CompositeReward Errors

| Error Condition | Handling | Log Message |
|----------------|----------|-------------|
| Component reward exception | Use 0.0 fallback | "WARNING: Reward {name} failed: {error}, using fallback 0.0" |

## Testing Strategy

### Testing Framework

- **Unit Testing**: pytest
- **Property-Based Testing**: hypothesis (Python PBT library)
- **Minimum iterations**: 100 per property test

### Unit Tests

1. **DiffSBDDModel Tests**
   - Test initialization with valid checkpoint and pocket
   - Test sample() returns correct structure
   - Test interface compatibility with EDMModel

2. **PocketLoader Tests**
   - Test loading RE-CmeB PDB file
   - Test error handling for missing files

3. **DockingReward Tests**
   - Test initialization with receptor
   - Test reward computation for known molecules
   - Test Ray parallelization

4. **Integration Tests**
   - Test full pipeline: model → reward → trainer
   - Test checkpoint save/load

### Property-Based Tests

Each correctness property will be implemented as a separate property-based test using hypothesis:

```python
# Example: Property 1 - Mask Conversion Round-Trip
@given(batch_sizes=st.integers(1, 8), node_counts=st.lists(st.integers(5, 50), min_size=1, max_size=8))
def test_mask_round_trip(batch_sizes, node_counts):
    """
    **Feature: diffsbdd-rlpf-integration, Property 1: Mask Conversion Round-Trip**
    **Validates: Requirements 2.3**
    """
    batch_mask = create_batch_mask(node_counts)
    node_mask = batch_mask_to_node_mask(batch_mask)
    recovered = node_mask_to_batch_mask(node_mask)
    assert torch.equal(batch_mask, recovered)
```

### Test File Organization

```
tests/
├── unit/
│   ├── test_diffsbdd_model.py
│   ├── test_mask_utils.py
│   ├── test_docking_reward.py
│   ├── test_druglikeness_reward.py
│   └── test_composite_reward.py
├── property/
│   ├── test_mask_properties.py      # Property 1
│   ├── test_pocket_properties.py    # Properties 2, 3
│   ├── test_reward_properties.py    # Properties 4, 5, 6, 7, 8, 9
│   ├── test_logprob_properties.py   # Properties 10, 11
│   └── test_model_properties.py     # Properties 12, 13
└── integration/
    └── test_training_pipeline.py
```

## File Structure

```
ddpo/
├── __init__.py
├── integration/
│   ├── __init__.py
│   ├── diffsbdd_model.py      # DiffSBDDModel class (Task 4)
│   └── mask_utils.py          # Mask conversion functions ✅
├── rewards/
│   ├── __init__.py
│   ├── base.py                # BaseReward interface ✅
│   ├── docking.py             # DockingReward class (Task 8)
│   ├── druglikeness.py        # DruglikenessReward class (Task 9)
│   └── composite.py           # CompositeReward class (Task 10)
├── utils/
│   ├── __init__.py
│   ├── molecule_converter.py  # Molecule format conversion (Task 6)
│   └── diffsbdd_filter.py     # DiffSBDDFilter - validity + build_molecule (Task 11)
├── configs/
│   ├── __init__.py
│   └── diffsbdd_ddpo.yaml     # Training configuration ✅
└── main_diffsbdd.py           # Entry point (Task 13)
```

**注意**: dataset_info 直接从 `DiffSBDD.constants.dataset_params['crossdock']` 获取，不创建单独文件。
