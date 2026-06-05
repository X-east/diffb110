# PMDM 99 分子生成测试报告

## 结论

PMDM 已完成源码、checkpoint、独立环境和生成测试。最终使用外部包装脚本复用 PMDM 官方模型、transform、construct 和 reconstruct 逻辑，未修改 PMDM 源码。测试设置为 100 diffusion steps、batch 32，在 RTX 3080 上峰值显存约 5281 MiB，比较接近本轮希望的 5000 MiB。

本轮目标为 100 个稳定单分子，实际尝试 288 个候选，得到 99 个稳定单分子；差 1 个是预生成 batch 数耗尽，不是 OOM 或崩溃。

## 环境与模型

| 项目 | 内容 |
|---|---|
| 源码 | `reference project/modern_sbdd_20260531/PMDM` |
| 环境 | `pmdm_modern_ref` |
| checkpoint | `analysis/modern_sbdd_20260531/assets/PMDM/500.pt` |
| 包装脚本 | `analysis/modern_sbdd_20260531/run_pmdm_for_pdb.py` |
| 测试 pocket | `reference project/modern_sbdd_20260531/DiffPharma/example/14gs.pdb` |
| 输出 SDF | `analysis/modern_sbdd_20260531/runs/pmdm_100_steps100_b32/output/pmdm_generated.sdf` |

## 运行设置

| 指标 | 数值 |
|---|---:|
| requested stable molecules | 100 |
| stable molecules obtained | 99 |
| attempted candidates | 288 |
| diffusion steps | 100 |
| batch size | 32 |
| num atoms | 24 |
| wall time | 190.59 s |
| GPU peak | 5281 MiB |
| unique stable SMILES | 99 |

## 环境修复记录

| 问题 | 处理 |
|---|---|
| 官方 `sample_for_pdb.py` 引用未定义 `args.savedir` | 用外部 wrapper 固定输出到 analysis |
| 克隆环境后 python 缺少可执行位 | 只修复 `pmdm_modern_ref` 内的 python 权限 |
| 缺 Biopython/einops | 在独立环境内补装 |
| PMDM 顶层 `utils` 与环境包名冲突 | wrapper 中注册 PMDM 本地 namespace |
| 新 RDKit 缺 `rdkit.six` | wrapper 中运行时兼容到 `six` |
| PyG 2.6.1 缺 `Inspector.distribute`，PyG 2.0.4 又缺 `_check_input` | 最终固定 `torch-geometric==2.3.1` |

## 99 分子性质统计

| 指标 | 数值 |
|---|---:|
| RDKit valid | 99 / 99 |
| unique SMILES | 99 |
| QED mean / median | 0.168 / 0.152 |
| logP mean / median | -2.071 / -2.239 |
| TPSA mean / median | 174.463 / 172.600 |
| MW mean / median | 380.750 / 377.332 |
| heavy atoms mean / median | 23.68 / 24 |
| hetero atoms mean / median | 14.19 / 14 |
| carbon fraction mean / median | 0.400 / 0.375 |
| ring count mean / median | 2.73 / 3 |
| aromatic ring rate | 0.020 |
| three-member ring rate | 0.404 |
| bad bond length fraction mean / median | 0.0170 / 0.0000 |

## 观察

PMDM 的速度和显存表现不错：100 steps 下 99 个稳定单分子约 3.18 分钟，batch 32 峰值 5.28GB，刚好落在本轮希望的显存区间。几何上比 SGEDiff 这次包装测试稳定很多，bad bond length 均值约 1.7%，中位数为 0。

但这批 100-step 输出的药化性质偏差很大：QED 低、logP 强烈偏负、TPSA 极高、杂原子数过多、芳环几乎没有、三元环比例 40.4%。这说明 PMDM 不能只看 valid/unique/速度；如果作为你的基础模型候选，需要重点检查原子类型 prior、采样步数、num_atom 设置，以及是否必须使用更接近官方 CrossDocked pocket 的输入/1000-step 设置。
