# Project Structure & Architecture

## 组件复用策略

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

### 复用 EDMActor 的影响

由于直接复用 RLPF 的 EDMActor，DiffSBDDModel 必须实现 EDMActor 调用的接口：

```python
# EDMActor.calculate_loss() 调用的方法
node_mask, edge_mask = self.model.get_mask(nodesxsample, batch_size, max_n_nodes)
_, log_prob, _, _ = self.model.sample_p_zs_given_zt(s, t, latents, node_mask, edge_mask, prev_sample=...)
```

这意味着：
- DiffSBDDModel 需要实现 `get_mask()` 返回 RLPF 格式的 mask `[B, N, 1]`
- DiffSBDDModel 需要实现 `sample_p_zs_given_zt()` 接受 RLPF 格式输入
- DiffSBDDModel 内部进行 RLPF ↔ DiffSBDD 格式转换

## Repository Layout

```
DiffB110/
├── DiffSBDD/                    # Base diffusion model (READ-ONLY)
│   ├── equivariant_diffusion/   # E(n)-equivariant diffusion implementation
│   │   ├── en_diffusion.py      # Core EnVariationalDiffusion class
│   │   ├── conditional_model.py # ConditionalDDPM - pocket-conditioned
│   │   ├── dynamics.py          # EGNNDynamics - dual encoder/decoder
│   │   └── egnn_new.py          # E(n) equivariant GNN layers
│   ├── lightning_modules.py     # LigandPocketDDPM - PyTorch Lightning
│   ├── analysis/                # Metrics, docking, visualization
│   ├── configs/                 # YAML configurations
│   └── generate_ligands.py      # Inference script
│
├── RLPF-verl_diffusion/         # RL Framework (SELECTED)
│   ├── verl_diffusion/
│   │   ├── model/
│   │   │   ├── base.py          # BaseModel interface ← Implement for DiffSBDD
│   │   │   └── edm_model.py     # EDMModel reference implementation
│   │   ├── trainer/
│   │   │   ├── ddpo_trainer.py  # DDPOTrainer - main training loop
│   │   │   └── config/          # Training configs
│   │   ├── worker/
│   │   │   ├── actor/edm_actor.py    # EDMActor - PPO updates
│   │   │   ├── reward/force.py       # ForceReward ← Replace with DockingReward
│   │   │   ├── rollout/edm_rollout.py # EDMRollout - trajectory sampling
│   │   │   └── filter/filter.py      # Sample filtering
│   │   ├── dataloader/          # Data loading utilities
│   │   └── protocol.py          # DataProto - data transfer protocol
│   └── main_edm.py              # Entry point
│
├── ddpo/                        # Integration layer (YOUR CODE)
│   ├── integration/
│   │   ├── diffsbdd_model.py    # DiffSBDDModel(BaseModel) wrapper
│   │   └── mask_utils.py        # Mask format conversion ✅
│   ├── rewards/
│   │   ├── base.py              # BaseReward interface ✅
│   │   ├── docking.py           # DockingReward - Vina + Ray parallel
│   │   ├── druglikeness.py      # QED, SA Score
│   │   └── composite.py         # Weighted combination
│   ├── utils/
│   │   ├── molecule_converter.py # Molecule format conversion
│   │   └── diffsbdd_filter.py   # DiffSBDDFilter (validity + build_molecule)
│   └── configs/
│       ├── diffsbdd_ddpo.yaml   # Training configuration ✅
│       └── dataset_info.py      # DiffSBDD crossdock dataset_info
│
├── data/
│   └── 8GJJ-C-Cip.pdb          # RE-CmeB protein target
│
├── tests/
│   ├── property/                # Property-based tests (hypothesis)
│   └── unit/                    # Unit tests
│
└── .kiro/
    ├── specs/                   # Feature specifications
    └── steering/                # AI assistant guidelines
```

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ddpo/ (Integration Layer)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐      ┌────────────────────────────┐   │
│  │ DiffSBDDModel        │      │ DockingReward              │   │
│  │ (BaseModel)          │      │ (BaseReward)               │   │
│  │                      │      │                            │   │
│  │ - sample()           │      │ @ray.remote                │   │
│  │ - sample_p_zs_given_zt()    │ def vina_score():          │   │
│  │ - compute_log_prob() │      │     return affinity        │   │
│  │ - cached_pocket      │      │                            │   │
│  └──────────┬───────────┘      └─────────────┬──────────────┘   │
│             │                                │                   │
│             ▼                                ▼                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 DDPOTrainer (RLPF)                        │   │
│  │                                                           │   │
│  │  rollout.generate_minibatch() → rewarder.calculate_rewards()│
│  │                    ↓                        ↓              │   │
│  │            DataProto(latents, logps)   DataProto(rewards) │   │
│  │                    ↓                        ↓              │   │
│  │              actor.update_policy(samples.union(rewards))  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DiffSBDD (Read-Only Reference)                │
│                                                                  │
│  ConditionalDDPM.sample_p_zs_given_zt(zt_lig, zt_pocket, ...)  │
│  EGNNDynamics.forward(xh_atoms, xh_residues, t, masks)          │
│  LigandPocketDDPM.prepare_pocket(residues)                      │
│  analysis.molecule_builder.build_molecule()                      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Integration Challenges & Solutions

### Challenge 1: Dual-Input Architecture
DiffSBDD uses `(ligand, pocket)` dual inputs, RLPF expects single input.

**Solution**: Cached pocket strategy
```python
class DiffSBDDModel(BaseModel):
    def __init__(self, checkpoint_path, pocket_pdb_path, device):
        self.model = LigandPocketDDPM.load_from_checkpoint(checkpoint_path)
        # Cache pocket at init time for performance
        self.pocket = self.model.prepare_pocket(residues)
```

### Challenge 2: Missing Log Probability
DiffSBDD's `sample_p_zs_given_zt` doesn't return log_prob.

**Solution**: Implement log_prob computation
```python
def compute_log_p_zs_given_zt(self, zs, mu, sigma, mask):
    log_prob = -0.5 * ((zs - mu)**2) / (sigma**2) - torch.log(sigma)
    return scatter_add(log_prob.sum(-1), mask, dim=0)
```

### Challenge 3: Mask Format Conversion
DiffSBDD uses `batch_mask` (node→batch), RLPF uses `node_mask` (B×N×1).

**Solution**: Conversion utilities in `ddpo/integration/mask_utils.py` ✅

### Challenge 4: h Format Handling (RESOLVED)
DiffSBDD returns `h` as tensor, RLPF EDMModel returns `h = {'integer': ..., 'categorical': ...}`.

**Key Finding**: RLPF Rollout only stores `h["categorical"]` to `batch["categorical"]`, discarding the dict structure. Downstream modules (Filter, Reward, Actor) read `batch["categorical"]` directly as tensor.

**Solution**: No conversion needed - return tensor directly
```python
# DiffSBDDModel.sample() returns tensor
h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)  # [B, N, 10]

# DiffSBDDRollout stores directly
batch_data = {
    "categorical": h_lig,  # tensor, not dict
}
```

### Challenge 5: build_molecule Incompatibility
RLPF uses QM9 `build_molecule`, DiffSBDD uses crossdock format.

**Solution**: Create DiffSBDDFilter using DiffSBDD's `analysis.molecule_builder.build_molecule`

## File Modification Guidelines

| Directory | Modification | Notes |
|-----------|-------------|-------|
| `DiffSBDD/` | ❌ Read-only | Reference only, do not modify |
| `RLPF-verl_diffusion/` | ❌ Read-only | Reference only, do not modify |
| `ddpo/` | ✅ Full control | Main development area |
| `data/` | ✅ Add data | Protein structures, results |
| `tests/` | ✅ Full control | Test files |

## SE(3) Equivariance Preservation

When integrating, ensure:
1. **Coordinate transforms** preserve equivariance (use DiffSBDD's `remove_mean_batch`)
2. **Policy networks** output equivariant actions (inherited from DiffSBDD)
3. **Rewards** are invariant (docking score doesn't change with rotation)
4. **Test**: `reward(R @ x) == reward(x)` for any rotation matrix R
