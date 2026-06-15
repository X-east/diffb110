# DecompOpt 测试报告

## Scope

- Repo: `reference project/modern_sbdd_20260531/DecompOpt`
- Env: `decompopt_modern_ref`
- Checkpoint: `analysis/modern_sbdd_20260531/assets/DecompOpt/decompopt.pt`
- Dataset: DecompOpt HF mirror full similarity LMDB + eval raw/test index
- Wrapper: `analysis/modern_sbdd_20260531/run_decomp_sample.py`
- Reference arms: `analysis/modern_sbdd_20260531/assets/DecompOpt/reference_arms_dataid004_top3.pt`

## Important Caveat

DecompOpt 官方流程不是单次 de novo 采样。它先用 DecompDiff 生成 arms，再用 Vina/minimize 选择 `best_mol_arms.pt`，然后 DecompOpt 用这些 arms 做优化采样。本次为了测试 checkpoint 和采样速度，用 LMDB 中自带的 retrieved arms 构造了 reference-arms 文件。因此结果可用于判断模型推理可运行性、速度和低/高步数质量差异，但不能等同于完整 DecompOpt 官方闭环性能。

## Runtime Tests

| run | requested | steps | batch | wall time | peak GPU MiB | reconstruct / complete |
|---|---:|---:|---:|---:|---:|---:|
| `decompopt_100_steps100_b20_refarms` | 100 | 100 | 20 | 399.13s | 2851 | 0 / 0 |
| `decompopt_20_steps1000_b20_refarms` | 20 | 1000 | 20 | 543.21s | 2849 | 19 / 19 |

## Molecule Summary

For the 1000-step run:

| metric | value |
|---|---:|
| SDF records | 19 |
| RDKit valid | 19 |
| unique smiles | 18 |
| unique rate valid | 0.947 |
| QED mean / median | 0.763 / 0.789 |
| logP mean / median | 2.633 / 2.728 |
| TPSA mean / median | 49.017 / 49.330 |
| carbon fraction mean | 0.839 |
| hetero atoms mean | 3.053 |
| aromatic ring rate | 0.579 |
| three-member ring rate | 0.000 |
| bad bond length fraction mean | 0.000 |

## Observations

1. DecompOpt is also highly sensitive to diffusion steps. At 100 steps it generated 100 raw attempts but reconstructed 0 molecules. At 1000 steps it reconstructed 19/20 complete molecules.
2. With retrieved-arms conditioning, the 1000-step molecules are more hydrophobic/aromatic and more lead-like than DecompDiff on the same data id: QED 0.763 vs DecompDiff 0.626, logP 2.63 vs 0.56, TPSA 49 vs 78.
3. Diversity is lower than DecompDiff in this small run. Unique rate was 18/19 and many molecules share a similar naphthalene/isopropylamine-like motif, consistent with strong arms conditioning.
4. Speed is slower than DecompDiff because DecompOpt adds retrieved-arm/fusion computation: 20 molecules at 1000 steps took 543.21s vs DecompDiff 404.85s on the same batch/data id.
5. GPU memory stayed around 2.85 GiB for batch 20; increasing batch size could better use a 5 GiB target, but the official script's reconstruction and model conditioning make step count a much larger bottleneck than memory here.

## Takeaway

DecompOpt is valuable to borrow for lead-optimization style generation: decomposed arms/scaffold priors plus external interaction/fitness-selected arms. It is less attractive as a direct replacement base generator if the workflow needs fast many-sample generation, because 100-step quality collapsed and 1000-step sampling is slow. A practical path for the user's project would be to borrow the conditioning design, not necessarily the whole inference loop.

## Files

- `runs/decompopt_100_steps100_b20_refarms/decompopt_100_steps100_refarms.sdf` (0 records)
- `runs/decompopt_20_steps1000_b20_refarms/decompopt_20_steps1000_refarms.sdf`
- `runs/decompopt_20_steps1000_b20_refarms/decompopt_20_steps1000_refarms_summary.json`
- `runs/decompopt_20_steps1000_b20_refarms/decompopt_20_steps1000_refarms_molecules.csv`
