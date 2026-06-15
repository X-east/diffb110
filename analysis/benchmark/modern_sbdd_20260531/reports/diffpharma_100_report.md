# DiffPharma 100 分子生成测试报告

## 结论

- 源码：`reference project/modern_sbdd_20260531/DiffPharma`
- 独立环境：`diffpharma_modern_ref`
- checkpoint：`analysis/modern_sbdd_20260531/assets/DiffPharma/DiffPharma_best.ckpt`
- checkpoint 来源：Zenodo `15428723`
- 输入：官方 example `14gs.pdb` + `14gs.sdf`
- 输出：`analysis/modern_sbdd_20260531/runs/diffpharma_100_b24/output`
- 规模：100 samples，batch size 24
- 状态：成功

## 速度与显存

| 指标 | 数值 |
|---|---:|
| wall time | 474.71 s |
| user time | 476.10 s |
| sys time | 7.99 s |
| SDF records | 100 |
| RDKit valid | 99 |
| 有效分子摊销 | 4.80 s/valid mol |
| GPU0 peak memory | 2257 MiB |

batch 24 仍只占约 2.26 GiB，说明这个入口和 pocket 下 batch 还可以继续提高。考虑到本轮 100 个已经用时接近 8 分钟，下一轮如专门测显存，可直接试 batch 48 或 64。

## 质量摘要

| 指标 | 数值 |
|---|---:|
| unique valid | 77 / 99 |
| QED mean | 0.544 |
| logP mean | 0.976 |
| TPSA mean | 77.96 |
| heavy atoms mean | 16.71 |
| carbon fraction mean | 0.733 |
| hetero atoms mean | 5.21 |
| aromatic ring rate | 83.8% |
| three-member ring rate | 12.1% |
| bad bond-length fraction mean | 0.0% |

## 观察

DiffPharma 比 DiffInt 慢很多，但分子结构分布更像药物：芳香环比例高、碳比例接近 ChEMBL 参考区间、三元环比例低得多。它的 interaction/pharmacophore 约束非常值得借鉴，尤其适合补你现在 docking/classifier 过度偏疏水、缺氢键/相互作用约束的问题。

## 文件

- `analysis/modern_sbdd_20260531/runs/diffpharma_100_b24/output/processed/14gs_gen.sdf`
- `analysis/modern_sbdd_20260531/runs/diffpharma_100_b24/diffpharma_summary.json`
- `analysis/modern_sbdd_20260531/runs/diffpharma_100_b24/diffpharma_molecules.csv`
- `analysis/modern_sbdd_20260531/runs/diffpharma_100_b24/gpu_mem.csv`
