# Apo2Mol 100 molecules test report

## Scope

- Project: `reference project/modern_sbdd_20260531/Apo2Mol`
- Environment: `apo2mol_modern_ref`
- Checkpoint: `reference project/modern_sbdd_20260531/Apo2Mol/apo2mol_dataset/apo2mol_checkpoint.ckpt`
- BAPNet checkpoint: `reference project/modern_sbdd_20260531/Apo2Mol/pretrained_models/PMINet`
- Wrapper: `analysis/modern_sbdd_20260531/run_apo2mol_single.py`
- Input used for this fallback test: DiffPharma `14gs.pdb` as both apo and holo pocket, plus `14gs.sdf` as reference ligand atom-count/template source
- Sampling: 100 requested molecules, batch 16, 100 diffusion steps
- Run directory: `analysis/modern_sbdd_20260531/runs/apo2mol_100_steps100_b16_synthetic14gs/`

## Data Availability Note

The public repository includes the pretrained checkpoint and split/index files, but not `apo2mol_dataset/data_folder`. I attempted to download the official Hugging Face dataset:

- Direct Hugging Face URL with proxy unset: timed out.
- Default proxy: failed because local proxy `127.0.0.1:7898` refused connection.
- `hf-mirror.com` dataset URL/API: returned 403/404 for this dataset path.

Therefore this is not an official Apo2Mol dataset evaluation. It is a checkpoint/runtime smoke and fallback generation test using a synthetic apo=holo 14gs data object.

## Runtime

| Item | Value |
|---|---:|
| Requested molecules | 100 |
| Batch size | 16 |
| Diffusion steps | 100 |
| Wall time | 225.15 s |
| Internal diffusion time | 215.74 s |
| Internal seconds/sample | 2.157 |
| Peak GPU memory | 5589 MiB |
| Mean sampled GPU memory | 5382.4 MiB |
| Reconstruct success | 16/100 |
| Single-component complete | 13/100 |

Batch-size attempts:

| Batch | Result |
|---:|---|
| 100 | OOM, tried to allocate 7.32 GiB with 7.86 GiB already allocated |
| 50 | OOM, tried to allocate 1.55 GiB with 6.48 GiB already allocated |
| 16 | Completed, peak 5589 MiB |

## Molecule Summary

The SDF contains the 16 molecules that survived reconstruction.

| Metric | Value |
|---|---:|
| SDF records | 16 |
| RDKit valid among written SDF | 16/16 |
| Unique SMILES among valid | 16/16 |
| QED mean / median | 0.175 / 0.136 |
| logP mean / median | 3.437 / 3.813 |
| TPSA mean / median | 94.548 / 88.450 |
| MW mean / median | 438.248 / 436.111 |
| Heavy atoms mean / median | 20.313 / 20.000 |
| Carbon fraction mean / median | 0.304 / 0.308 |
| Hetero atoms mean / median | 14.125 / 14.000 |
| HBA mean / HBD mean | 7.125 / 5.688 |
| Ring count mean | 0.938 |
| Aromatic-ring molecule rate | 0.000 |
| Three-member-ring molecule rate | 0.000 |
| Bad bond length fraction mean | 0.531 |

## Observations

- The checkpoint and BAPNet condition model load successfully, and the diffusion loop runs on GPU.
- Memory behavior is heavy: the BAPNet condition module creates dense pocket graph edges, so batch size is limited. Batch 16 is the largest tested setting that completed and it used about 5.6 GB peak GPU memory.
- The fallback synthetic input is chemically unreliable. Reconstruction success is only 16/100, and the reconstructed molecules have very high bad-bond-length fraction, low QED, no aromatic rings, and excessive hetero/halogen-heavy chemistry.
- These numbers should not be used to judge Apo2Mol's paper-level quality. They mainly show that the public checkpoint can be loaded and run, but the official dataset is required for a fair generation test.

## Takeaway For DiffB110

Apo2Mol is conceptually important if RE-CmeB pocket flexibility matters, because it jointly models ligand generation and holo-like pocket motion. For immediate DiffB110 integration, it is not the best short-term base-model replacement: the public data dependency is large and currently unreachable from this machine, inference is slower/heavier than DiffSBDD/DiffGui-style sampling, and the fallback synthetic test produced poor chemistry. The useful idea to borrow is dynamic pocket conditioning, not the exact runtime stack.
