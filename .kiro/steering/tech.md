# Technology Stack & Build System

## Runtime Environment

- **OS**: CentOS 7
- **Conda Environment**: `dif` (actual environment name)
- **Shell**: bash
- **OpenBabel Plugin Path**: `~/miniforge3/envs/dif/lib/openbabel/3.1.0`
- **Search command caveat**: do not use `rg` / `rg --files` on this machine.
  The installed ripgrep can fail through snap permission checks; use `grep`,
  `find`, `sed`, or `ls` instead.

## Version Priority Rule

When dependencies exist in multiple projects, follow this priority:
1. **DiffSBDD version** (highest priority) - Base model compatibility
2. **RLPF version** (second priority) - RL framework compatibility  
3. **Latest stable version** (fallback) - Only if not specified in either

## Core Frameworks

### Deep Learning
- **PyTorch 2.0.1** - From DiffSBDD environment.yaml
- **PyTorch Lightning 1.8.4** - From DiffSBDD environment.yaml
- **CUDA 11.8** - From DiffSBDD environment.yaml

### RL Framework: RLPF (verl_diffusion)
- **Location**: `RLPF-verl_diffusion/`
- **Key Modules**:
  - `DDPOTrainer` - Main DDPO training loop
  - `EDMModel` - Diffusion model wrapper (adapt for DiffSBDD)
  - `EDMActor` - PPO policy gradient updates
  - `EDMRollout` - Trajectory sampling
  - `ForceReward` → Replace with `DockingReward`
  - `DataProto` - Standardized data transfer protocol
- **Features**: Ray parallel rewards, wandb logging, checkpoint management

### Cheminformatics
- **RDKit 2022.03+** - Molecular processing, validity checks, property calculation
- **Smina** - Docking score computation (primary reward, installed separately)
- **OpenBabel 3.1** - Molecular format conversion (plugin path: `~/miniforge3/envs/dif/lib/openbabel/3.1.0`)
- **Uni-Mol** - Universal 3D molecular representation learning
  - **Version**: Latest from PyPI (`pip install unimol_tools`)
  - **Purpose**: High-dimensional feature extraction for discriminator
  - **Input**: SMILES strings (automatically generates 3D conformers internally)
  - **Output**: 512-dim molecular representations
  - **Location**: `/home/md666/data/disk6/Uni-Mol-main` (reference implementation)

### Molecular Format Conversion Strategy

**Key Decision**: Use SMILES as the unified format for discriminator input.

**Rationale**:
1. **Avoid conformer bias**: Generated molecules (binding pose) vs database drugs (global minimum) are not directly comparable in 3D
2. **Uni-Mol handles 3D internally**: Accepts SMILES, automatically generates consistent 3D conformers for feature extraction
3. **Discriminator focuses on chemistry**: Judges "is this a drug-like molecule?" not "is this a good binding pose?"
4. **Simplified pipeline**: Database drugs are typically stored as SMILES, no extra processing needed

**Conversion Pipeline**:
```
Generated 3D molecule (x, h)
    ↓
coords_to_smiles() → SMILES string
    ↓
Uni-Mol: SMILES → auto-generate 3D → 512-dim embedding
    ↓
Discriminator: [PoseBusters(18) + ECFP4(2048) + Uni-Mol(512)] → 2578-dim features
```

**Implementation**:
- `ddpo/utils/molecule_converter.py`: 3D coords ↔ RDKit Mol ↔ SMILES ↔ PDBQT
- `ddpo/rewards/manifold_discriminator.py`: Uni-Mol integration for feature extraction
- See `UNIMOL_INTEGRATION.md` for detailed design decisions

### Configuration & Logging
- **Hydra** - Configuration management
- **Weights & Biases (wandb)** - Experiment tracking
- **Ray** - Parallel reward computation (critical for docking)

## Environment Setup

### Conda Environment
```bash
# Activate existing environment
conda activate dif

# Set OpenBabel plugin directory (required for molecule conversion)
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Install Uni-Mol for molecular representation
pip install unimol_tools --upgrade
pip install huggingface_hub  # For automatic model downloads

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import rdkit; print(f'RDKit: {rdkit.__version__}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
python -c "from unimol_tools import UniMolRepr; print('Uni-Mol OK')"
```

### Key Dependencies
```
pytorch>=2.0.1
pytorch-lightning>=1.8.4
rdkit>=2022.03.2
biopython>=1.79
pytorch-scatter>=2.1.2
wandb>=0.13.1
ray>=2.41.0
tensordict>=0.2.0
unimol-tools>=0.1.3  # Uni-Mol molecular representation
huggingface-hub      # For Uni-Mol model downloads
```

## Common Commands

### DiffSBDD Baseline
```bash
# Activate environment
conda activate dif
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Training
python DiffSBDD/train.py --config DiffSBDD/configs/crossdock_ca_cond.yml

# Generate ligands for RE-CmeB
python DiffSBDD/generate_ligands.py checkpoints/model.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb --n_samples 100
```

### RLPF Training
```bash
# Activate environment
conda activate dif
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Start DDPO training
python ddpo/main_diffsbdd.py --config ddpo/configs/production_training.yaml

# CPU parallel test (500 steps, Ray parallel docking)
bash scripts/run_cpu_parallel_test.sh
```

### Docking Evaluation
```bash
# Smina scoring (installed separately)
smina --receptor data/8GJJ-C-Cip.pdb --ligand ligand.sdf \
    --scoring vinardo --score_only
```

## Project Data Paths
- **Protein target**: `data/8GJJ-C-Cip.pdb` (RE-CmeB)
- **DiffSBDD checkpoints**: `DiffSBDD/data/checkpoints/`
- **RLPF pretrained**: `RLPF-verl_diffusion/exp/edm-xtb/`
- **Training outputs**: `./exp/diffsbdd-ddpo/`

## Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended
- **RAM**: 32GB+ for Ray parallel reward computation
- **CPU**: 8+ cores for parallel docking (Ray workers)
- **Storage**: 50GB+ for datasets and checkpoints

## Performance Notes
- Docking with Vina: ~3-5 seconds per molecule
- Ray parallelization (8 workers): ~8x speedup
- Batch size 128: ~80s with parallel vs ~640s serial
