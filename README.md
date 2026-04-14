# DiffSBDD-RLPF: Protein-Specific Molecule Generation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **Optimizing DiffSBDD for RE-CmeB protein binding via DDPO**

---

## Project Overview

This project integrates DiffSBDD (E(n)-Equivariant Diffusion Model) with the RLPF framework to optimize molecular generation for the RE-CmeB protein target using reinforcement learning.

### Key Features

- Protein-Specific Optimization - Docking-based rewards for RE-CmeB (PDB: 8GJJ)
- SE(3) Equivariance - Preserves DiffSBDD's geometric properties
- Ray Parallel Docking - Efficient AutoDock Vina scoring
- Multi-Objective Rewards - Binding affinity + QED + SA Score

---

## Project Structure

```
DiffB110/
├── ddpo/                        # Integration layer
│   ├── integration/             # DiffSBDDModel, mask_utils
│   ├── rewards/                 # Docking, druglikeness, composite
│   ├── utils/                   # Molecule converter, filter
│   └── configs/                 # Training configuration
├── DiffSBDD/                    # Base model (READ-ONLY)
├── RLPF-verl_diffusion/         # RL framework reference
├── data/                        # RE-CmeB protein (8GJJ)
└── tests/                       # Unit and property tests
```

---

## Development Progress

**Completed**: Tasks 1-3 (infrastructure, mask utils, pocket loading)

**Next**: Task 4 (dataset_info configuration)

See `.kiro/specs/diffsbdd-rlpf-integration/tasks.md` for full plan.

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Docking Score | < -8 kcal/mol |
| Molecular Validity | > 80% |
| QED | > 0.5 |
| SA Score | < 4 |

---

**Last Updated**: 2024-12-18
