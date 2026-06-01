# GCDM-SBDD 100 molecules test report

## Scope

- Project: `reference project/modern_sbdd_20260531/GCDM-SBDD`
- Environment: `gcdm_sbdd_modern_ref`
- Checkpoint: `analysis/modern_sbdd_20260531/assets/GCDM-SBDD/extracted/checkpoints/crossdocked_ca_cond_gcpnet.ckpt`
- Pocket input: `reference project/modern_sbdd_20260531/DiffPharma/example/14gs.pdb`
- Pocket definition: 32 CA residues from the 14gs pocket residue list
- Sampling: requested 100 molecules, fixed `num_nodes_lig=24`, `timesteps=100`, `--sanitize`
- Run directory: `analysis/modern_sbdd_20260531/runs/gcdm_100_steps100_crossdock_gcpnet/`

## Runtime

| Item | Value |
|---|---:|
| Wall time | 117.08 s |
| Peak GPU memory | 5049 MiB |
| Mean sampled GPU memory | 4618.1 MiB |
| GPU memory samples | 105 |
| Output SDF records | 45 |

## Molecule Summary

Because `--sanitize` is enabled, GCDM drops molecules that fail RDKit sanitization before writing the SDF. Therefore the practical valid yield is 45/100 requested molecules.

| Metric | Value |
|---|---:|
| Sanitized output yield | 45/100 |
| RDKit valid among written SDF | 45/45 |
| Unique SMILES among valid | 45/45 |
| QED mean / median | 0.517 / 0.520 |
| logP mean / median | 1.300 / 1.497 |
| TPSA mean / median | 78.989 / 76.630 |
| MW mean / median | 289.500 / 320.485 |
| Heavy atoms mean / median | 19.844 / 23.000 |
| Carbon fraction mean / median | 0.716 / 0.708 |
| Hetero atoms mean / median | 5.756 / 6.000 |
| HBA mean / HBD mean | 4.511 / 3.467 |
| Ring count mean | 2.044 |
| Aromatic-ring molecule rate | 0.022 |
| Three-member-ring molecule rate | 0.067 |
| Bad bond length fraction mean | 0.000 |

## Observations

- Speed and memory are in the desired test band: a 100-molecule run used about 5.0 GB peak GPU memory and completed in about 117 s.
- Geometry among written molecules is clean by the current bond-length checker, with mean bad-bond fraction 0.000.
- Chemical validity before sanitization is the main weakness in this setting. The log contains many explicit-valence failures, and only 45 of 100 requested molecules survived `--sanitize`.
- The written valid molecules are diverse and chemically less pathological than PMDM in this same quick-test setting: QED is moderate, logP/TPSA are reasonable, and the carbon/hetero split is not collapsed.
- Aromatic chemistry is underrepresented in this 100-step sample, with only 2.2% of valid molecules containing aromatic rings. That may be partly due to the fixed 24-node setting, 100-step truncation, and using a generic 14gs pocket outside the original benchmark data pipeline.

## Takeaway For DiffB110

GCDM-SBDD is worth borrowing for geometry-complete modeling ideas and pocket representation, especially if the goal is to improve bond-length and 3D plausibility beyond DiffSBDD's atom-cloud plus rule-bond reconstruction. As a direct base-model replacement, this quick run is not yet strong enough without more tuning: the sanitized valid yield is too low at 100 steps on this pocket. It is still a useful reference implementation because it reached the requested memory scale, generated at moderate speed, and produced clean geometry for the molecules that survived sanitization.
