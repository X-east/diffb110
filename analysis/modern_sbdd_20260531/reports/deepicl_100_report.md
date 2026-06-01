# DeepICL 100 molecules test report

## Scope

- Project: `reference project/modern_sbdd_20260531/DeepICL`
- Environment: `deepicl_modern_ref`
- Checkpoint: `reference project/modern_sbdd_20260531/DeepICL/script/best_model/save_best.pt`
- Input: official example `example/2f0z_protein.pdb` plus `example/2f0z_ligand.sdf`
- Preprocessing wrapper: `analysis/modern_sbdd_20260531/prepare_deepicl_example.py`
- Sampling mode: pocket-only heuristic interaction condition, 100 samples, `max_num_add_atom=30`
- Run directory: `analysis/modern_sbdd_20260531/runs/deepicl_2f0z_100_cpu/`
- Aggregate SDF: `analysis/modern_sbdd_20260531/runs/deepicl_2f0z_100_cpu/deepicl_aggregate_100.sdf`

## Runtime

DeepICL's official generation script is CPU-multiprocessing oriented. In this code path, `--ngpu=1` does not select CUDA, while `--ngpu>1` can create device mismatch because model/data movement is incomplete. I therefore tested the intended CPU path.

| Item | Value |
|---|---:|
| CPU workers | 18 |
| Wall time | 75.07 s |
| User CPU time | 808.70 s |
| System CPU time | 198.30 s |
| Output SDF files | 100 |
| GPU memory | not applicable |

## Molecule Summary

| Metric | Value |
|---|---:|
| SDF records | 100 |
| RDKit valid | 100/100 |
| Unique SMILES | 100/100 |
| QED mean / median | 0.567 / 0.579 |
| logP mean / median | -0.328 / -0.396 |
| TPSA mean / median | 104.169 / 103.690 |
| MW mean / median | 332.104 / 348.210 |
| Heavy atoms mean / median | 25.740 / 27.000 |
| Carbon fraction mean / median | 0.729 / 0.733 |
| Hetero atoms mean / median | 6.820 / 6.500 |
| HBA mean / HBD mean | 3.560 / 0.000 |
| Ring count mean | 2.510 |
| Aromatic-ring molecule rate | 0.690 |
| Three-member-ring molecule rate | 0.020 |
| Bad bond length fraction mean | 0.000 |

## Observations

- This is not a diffusion baseline, but an interaction-conditioned autoregressive generator. It is useful as a design reference rather than a direct replacement for DiffSBDD.
- The quick test produced excellent formal validity and uniqueness: 100/100 valid and 100/100 unique.
- The molecules are moderately drug-like by QED, have reasonable ring/aromatic content, and show very low three-member-ring rate.
- The generated molecules are quite polar in this pocket-only setup, with logP around -0.33 and TPSA around 104. HBD is always zero in this run, which suggests the atom/bond reconstruction and model preferences should be checked before using it as a chemical-quality oracle.
- The official script's GPU handling is incomplete for this local run, so it cannot satisfy the 5 GB GPU-load target without source modification. Since the user requested no project-code changes, I kept the official CPU path.

## Takeaway For DiffB110

DeepICL is one of the better conceptual references for adding explicit interaction priors. The main thing to borrow is not its runtime stack, but the conditioning design: pocket atoms carry interaction labels such as HBA/HBD/hydrophobic/aromatic/salt terms, and generation is steered locally by these interaction features. This maps well to the DiffB110 pain point where docking and classifier rewards can arrive too late. A practical next step would be to add interaction-condition features or reward shaping to DiffSBDD/DiffGui-style generation rather than replacing the base model with DeepICL directly.
