# DiffInt 100 分子生成测试报告

## 结论

- 源码：`reference project/modern_sbdd_20260531/DiffInt`
- 独立环境：`diffint_modern_ref`
- checkpoint：`reference project/modern_sbdd_20260531/DiffInt/checkpoints/best_model.ckpt`
- 输入：官方 example `1a2g_A_rec.pdb` + `1a2g_A_rec.sdf`
- 输出：`analysis/modern_sbdd_20260531/runs/diffint_100/output`
- 规模：默认入口生成 100 个分子，默认 batch 120
- 状态：成功

## 速度与显存

| 指标 | 数值 |
|---|---:|
| wall time | 42.45 s |
| user time | 43.32 s |
| sys time | 7.20 s |
| SDF records | 100 |
| RDKit valid | 99 |
| 有效分子摊销 | 0.43 s/valid mol |
| GPU0 peak memory | 3188 MiB |

显存峰值没有达到用户希望的约 5000 MiB，说明这个 example pocket 下 batch 120 仍偏保守；后续同类模型可把 batch 提高到 180-220 再试。

## 质量摘要

| 指标 | 数值 |
|---|---:|
| unique valid | 99 / 99 |
| QED mean | 0.597 |
| logP mean | 1.650 |
| TPSA mean | 51.06 |
| heavy atoms mean | 16.27 |
| carbon fraction mean | 0.735 |
| hetero atoms mean | 3.88 |
| aromatic ring rate | 25.3% |
| three-member ring rate | 31.3% |
| bad bond-length fraction mean | 0.0% |

## 观察

DiffInt 速度很好，100 个分子 42.45 s，显存也比较轻。键长粗指标很好，但芳香环比例偏低、三元环比例偏高；它更值得借鉴的是显式 interaction/hydrogen-bond 条件，而不是直接作为最终高质量生成器。

## 文件

- `analysis/modern_sbdd_20260531/runs/diffint_100/output/processed/1a2g_A_rec_gen.sdf`
- `analysis/modern_sbdd_20260531/runs/diffint_100/diffint_summary.json`
- `analysis/modern_sbdd_20260531/runs/diffint_100/diffint_molecules.csv`
- `analysis/modern_sbdd_20260531/runs/diffint_100/gpu_mem.csv`
