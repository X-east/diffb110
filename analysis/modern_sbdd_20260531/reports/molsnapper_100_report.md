# MolSnapper 100 分子生成测试报告

## 结论

MolSnapper 已完成源码、checkpoint、独立环境和生成测试。最终未改源码，采用两个独立 50 分子进程规避同一进程循环内显存未充分释放的问题，共聚合前 100 个 shifted SDF 分子做统计。

## 环境与模型

| 项目 | 内容 |
|---|---|
| 源码 | `reference project/modern_sbdd_20260531/MolSnapper` |
| 环境 | `molsnapper_modern_ref` |
| diffusion checkpoint | `reference project/modern_sbdd_20260531/MolSnapper/ckpt/MolDiff.pt` |
| bond checkpoint | `reference project/modern_sbdd_20260531/MolSnapper/ckpt/bondpred.pt` |
| 测试 pocket | `data/example_1h00/processed_pocket_1h00.pkl` |
| 条件 SDF | `data/example_1h00/ref_points.sdf` |
| 输出聚合 SDF | `analysis/modern_sbdd_20260531/runs/molsnapper_2x50_b64/molsnapper_aggregate_100_shifted.sdf` |

## 运行设置

| run | seed | config | batch | 生成成功数 | wall time | GPU peak |
|---|---:|---|---:|---:|---:|---:|
| chunk 1 | 2023 | `molsnapper_sample_50_seed2023.yml` | 64 | 63 | 255.41 s | 9657 MiB |
| chunk 2 | 2024 | `molsnapper_sample_50_seed2024.yml` | 64 | 60 | 259.69 s | 9655 MiB |

补充试探：

| batch | 结果 |
|---:|---|
| 256 | OOM，bond predictor dense edge block 爆显存 |
| 128 | OOM，同样在 bond predictor |
| 64 单进程 100 分子 | 第一轮生成 62 个，第二轮 OOM；因此最终拆成两个独立进程 |

## 100 分子性质统计

| 指标 | 数值 |
|---|---:|
| SDF records | 100 |
| RDKit valid | 100 / 100 |
| unique SMILES | 98 |
| unique rate valid | 0.980 |
| QED mean / median | 0.546 / 0.527 |
| logP mean / median | 3.688 / 3.669 |
| TPSA mean / median | 56.880 / 56.465 |
| MW mean / median | 316.647 / 313.856 |
| heavy atoms mean / median | 23.08 / 23 |
| hetero atoms mean / median | 4.58 / 5 |
| carbon fraction mean / median | 0.805 / 0.800 |
| ring count mean / median | 3.36 / 3 |
| aromatic ring count mean / median | 2.43 / 3 |
| has aromatic ring rate | 0.930 |
| three-member ring rate | 0.180 |
| bad bond length fraction mean / median | 0.0050 / 0.0000 |

## 观察

MolSnapper 的显式 bond predictor 让键级与几何统计明显比纯规则补键路线更值得借鉴，聚合 100 个分子 RDKit 全有效、bad bond length 均值约 0.5%。但工程代价也很清楚：在 RTX 3080 10GB 上 batch 64 已经接近满显存，batch 128/256 不可用，并且同一进程多轮采样存在显存累积/OOM 风险。

对你的项目借鉴价值主要在三点：第一，生成端直接预测 bond/half-edge，比 DiffSBDD 后处理推断键更适合解决键长与成键一致性；第二，pharmacophore/ref-points 条件化接口适合接你的已知相互作用假设；第三，若要放进强化学习，采样吞吐和显存生命周期需要先重构，否则 rollout 过程会非常吃卡。
