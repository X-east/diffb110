# DiffSBDD-RL: Protein-Specific Molecule Generation

## Project Goal
Optimize DiffSBDD (E(n)-Equivariant Diffusion Model) for generating drug-like molecules with high binding affinity to the **RE-CmeB protein** using reinforcement learning.

## Strategy
**Minimal Engineering Approach**: Interface DiffSBDD with RLPF (verl_diffusion) framework rather than building DDPO from scratch.

## Key Components
- **Base Model**: DiffSBDD - 3D structure-based drug design with SE(3) equivariance
- **RL Framework**: RLPF (verl_diffusion) - DDPO training with Ray parallel rewards
- **Target**: RE-CmeB protein (PDB: 8GJJ) binding optimization
- **Rewards**: Docking scores (AutoDock Vina), QED, SA Score, validity

## Core Workflow
1. DiffSBDD generates 3D molecular structures conditioned on fixed RE-CmeB pocket
2. RLPF rollout collects trajectories with log probabilities
3. DockingReward evaluates binding affinity via Ray parallel Vina scoring
4. EDMActor performs PPO policy gradient updates

## Key Integration Points
- `DiffSBDDModel(BaseModel)`: Wrapper adapting DiffSBDD to RLPF interface
- `DockingReward(BaseReward)`: AutoDock Vina with Ray parallelization
- Fixed pocket strategy: RE-CmeB pocket as constant condition

## Success Metrics
- Improved docking scores for RE-CmeB target (< -8 kcal/mol)
- Maintained molecular validity (> 80%)
- Drug-likeness: QED > 0.5, SA < 4
- Preserved SE(3) equivariance properties
