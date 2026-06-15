# SGEDiff 100 分子生成测试报告

## 结论

SGEDiff 已完成源码、checkpoint、独立环境和一次可复现生成测试。由于仓库没有提供 README 中硬编码的 `data/Test/...` 示例，也没有提供 `target_100.pdata` 这类官方已处理测试集，本次采用不改源码的外部包装脚本，把 DiffPharma 的 `14gs` pocket/ligand 构造成 SGEDiff 所需 `Data` 后调用官方模型与采样函数。

这次测试能衡量速度和工程可运行性，但生成质量不能视为官方公平结果：100 个模型输出中只有 7 个可通过 SGEDiff/OpenBabel/RDKit 重建为非空合法分子，且化学组成明显异常。

## 环境与模型

| 项目 | 内容 |
|---|---|
| 源码 | `reference project/modern_sbdd_20260531/SGEDiff` |
| 环境 | `sgediff_modern_ref` |
| checkpoint | `trained_model/SGEDiff/1000000.pt` |
| 包装脚本 | `analysis/modern_sbdd_20260531/run_sgediff_single.py` |
| 测试 config | `analysis/modern_sbdd_20260531/sgediff_sample_100_steps100.yml` |
| pocket/ligand | `reference project/modern_sbdd_20260531/DiffPharma/example/14gs.pdb` / `14gs.sdf` |
| 输出目录 | `analysis/modern_sbdd_20260531/runs/sgediff_100_steps100_b16/output` |

## 运行设置

| 指标 | 数值 |
|---|---:|
| sampling model | SGEDiff-NG |
| requested samples | 100 |
| diffusion steps | 100 |
| batch size | 16 |
| wall time | 249.44 s |
| GPU peak | 2073 MiB |
| model raw outputs | 100 |
| reconstructed non-empty SDF | 7 |

## 7 个可重建分子的性质统计

| 指标 | 数值 |
|---|---:|
| RDKit valid SDF records | 7 |
| unique SMILES | 7 |
| QED mean / median | 0.295 / 0.162 |
| logP mean / median | 2.183 / 1.581 |
| TPSA mean / median | 77.667 / 75.430 |
| MW mean / median | 338.208 / 366.203 |
| heavy atoms mean / median | 15.43 / 16 |
| hetero atoms mean / median | 11.29 / 12 |
| carbon fraction mean / median | 0.259 / 0.294 |
| aromatic ring rate | 0.000 |
| three-member ring rate | 0.000 |
| bad bond length fraction mean / median | 0.414 / 0.438 |

## 关键观察

SGEDiff 的 subgraph-enriched 思路值得借鉴，但这个公开仓库的可复现实验链条不完整：README 的采样入口硬编码缺失数据路径，预处理依赖 PyMOL/PLIP/PuresNet/Vina，仓库没有提供官方 processed test data。直接绕开官方预处理后，模型虽然能在低显存下快速跑完，但原子类型分布明显坍到 P/S/F/Cl/O/N 这类异常组合，碳比例只有 0.259，平均坏键长比例高达 41.4%，不适合作为当前可靠 baseline。

对你的项目，SGEDiff 更适合作为“subgraph/interaction 条件如何接入扩散模型”的结构参考，而不是直接替换 DiffSBDD 的现成基础模型。若后续要严肃评估，需要先拿到作者的 processed test set 或完整复现其 PyMOL/PLIP/PuresNet 预处理链，再跑 1000 step 官方设置。
