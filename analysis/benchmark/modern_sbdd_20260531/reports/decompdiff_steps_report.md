# DecompDiff 测试报告

## Scope

- Repo: `reference project/modern_sbdd_20260531/DecompOpt`
- Env: `decompdiff_modern_ref`
- Checkpoint: `analysis/modern_sbdd_20260531/assets/DecompOpt/decompdiff.pt`
- Dataset: DecompOpt HF mirror 的 full similarity LMDB + eval raw/test index
- Wrapper: `analysis/modern_sbdd_20260531/run_decomp_sample.py`
- 注意：这里测试的是 DecompDiff checkpoint，源码使用 DecompOpt 仓库中兼容的采样脚本。

## Assets

- `crossdocked_v1.1_rmsd1.0_processed_full_similarity.lmdb`: 5.0 GiB, downloaded from hf-mirror
- `crossdocked_v1.1_rmsd1.0_processed_full_similarity_name2id.pt`
- `split_by_name.pt`
- `test_index.pkl`
- eval raw/processed tar extracted under `analysis/modern_sbdd_20260531/assets/DecompOpt/extracted/data`

## Runtime Tests

| run | requested | steps | reconstruction | wall time | peak GPU MiB | output |
|---|---:|---:|---|---:|---:|---|
| `decompdiff_100_steps100_b20` | 100 | 100 | official bond recon | timed out at 1800s | 6613 | no final `result.pt`; reconstruction was still failing |
| `decompdiff_100_steps100_b20_nobondrecon` | 100 | 100 | distance recon fallback | 226.31s | 2817 | 8 SDF records |
| `decompdiff_20_steps1000_b20_bondrecon` | 20 | 1000 | official bond recon | 404.85s | 2817 | 19 SDF records |

## Molecule Summary

| run | SDF records | reconstruct / complete | QED mean | logP mean | TPSA mean | carbon frac | aromatic ring rate | bad bond length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 steps, no-bond recon | 8 | 8 / 0 | 0.197 | 2.451 | 100.334 | 0.303 | 0.000 | 0.326 |
| 1000 steps, bond recon | 19 | 19 / 19 | 0.626 | 0.562 | 77.972 | 0.709 | 0.368 | 0.000 |

## Observations

1. 100-step DecompDiff is not usable in this setup. With official bond reconstruction it mostly failed and got stuck in slow reconstruction; with distance reconstruction it produced only 8 written molecules out of 100, all fragmented (`n_complete=0`) and chemically poor.
2. 1000-step DecompDiff recovers normal-looking molecules: 19/20 reconstructed, 19/20 complete, unique rate 1.0, QED mean 0.626, bad bond length fraction 0.
3. The cost-quality tradeoff is sharp. On this data id, 1000 steps for 20 molecules took 404.85s, while 100 steps for 100 molecules took 226.31s but was effectively low quality. Extrapolating directly from this run, 1000 steps for 100 molecules at the same batch/data id would be about 33-34 minutes on one GPU.
4. Peak GPU memory was only 2817 MiB for batch 20 in the successful runs, lower than the user's desired ~5000 MiB. The earlier timed-out run showed 6613 MiB, but that included long official bond reconstruction failure behavior and should not be treated as a clean sampling memory target.

## Takeaway

DecompDiff is valuable as a design reference for decomposed scaffold/arm priors plus explicit bond diffusion, but it is not a fast drop-in replacement if high quality requires 1000 diffusion steps. For the user's goal, the most useful lesson is not "use DecompDiff directly"; it is to borrow its decomposition prior and bond-head training idea while keeping a faster base model or distilling/step-reducing a newer model.

## Files

- `runs/decompdiff_100_steps100_b20_nobondrecon/decompdiff_100_steps100_nobondrecon.sdf`
- `runs/decompdiff_100_steps100_b20_nobondrecon/decompdiff_100_steps100_nobondrecon_summary.json`
- `runs/decompdiff_20_steps1000_b20_bondrecon/decompdiff_20_steps1000_bondrecon.sdf`
- `runs/decompdiff_20_steps1000_b20_bondrecon/decompdiff_20_steps1000_bondrecon_summary.json`
