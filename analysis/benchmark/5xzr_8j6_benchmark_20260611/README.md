# 5XZR / 8J6_A_601 多模型生成与 UniDock 对接准备

## 输入确认

- 原始 PDB：`data/5xzr.pdb`
- 配体：`8J6 A 601`
- HETATM 统计：`8J6 A 601` 共 28 个重原子；另有两个水分子 `HOH A 701/702`
- 8A pocket：48 个残基，419 个蛋白重原子
- 配体 SDF：28 atoms / 31 bonds

## 关键文件

| 文件 | 作用 |
|---|---|
| `inputs/5XZR_8J6_protein_only.pdb` | 去除配体/水后的 protein-only PDB |
| `inputs/5XZR_8J6_pocket8A.pdb` | 以 8J6 为中心的 8A pocket |
| `inputs/5XZR_8J6_ligand.pdb` | native ligand PDB |
| `inputs/5XZR_8J6_ligand.sdf` | native ligand SDF |
| `inputs/5XZR_8J6_ligand_rdkit_fixed.sdf` | DiffInt/DiffPharma 专用参考配体 SDF；修正环内 N 电荷以通过 RDKit sanitize |
| `inputs/5XZR_8J6_pocket_residues_8A.txt` | DiffSBDD/DePPA residue list |
| `inputs/deppa_5xzr_test_pocket/` | DePPA 专用单 pocket 输入 |
| `inputs/5XZR_8J6_pocket8A.pdbqt` | UniDock receptor PDBQT |

## UniDock box

由 native 8J6 ligand 几何中心确定：

```text
center_x = -21.484
center_y = 70.592
center_z = 21.081
size_x = 20.0
size_y = 20.0
size_z = 20.0
```

native ligand 回对接 sanity check：

- output：`docking/native_8J6_unidock_fast_gpu_charged/`
- success：1/1
- UniDock score：`-12.737`

## 正式生成命令

一键正式 pipeline（核心三模型 -> 补充三模型 -> UniDock -> summary）：

```bash
TOTAL=500 TIMESTEPS=500 GPUS="0 1 2 3 4 5 6" BATCH_SIZE=128 \
  nohup bash analysis/5xzr_8j6_benchmark_20260611/launch_full_pipeline.sh \
  > analysis/5xzr_8j6_benchmark_20260611/logs/full_pipeline.log 2>&1 &
```

完成后重点看：

- `aggregated/*.sdf`
- `evaluation/six_models/model_summary.csv`
- `docking/unidock_fast_5xzr_8j6/*/run_summary.json`
- `summary/benchmark_summary.csv`
- `summary/benchmark_docking_scores.png` 和 `summary/benchmark_quality_summary.png`

也可以分段运行。

核心三模型：

```bash
TOTAL=500 TIMESTEPS=500 GPUS="0 1 2 3 4 5 6" \
  nohup bash analysis/5xzr_8j6_benchmark_20260611/launch_core_benchmark.sh \
  > analysis/5xzr_8j6_benchmark_20260611/logs/launch_core.log 2>&1 &
```

补充三模型：

```bash
TOTAL=500 TIMESTEPS=500 GPUS="0 1 2 3 4 5 6" \
  nohup bash analysis/5xzr_8j6_benchmark_20260611/launch_second_benchmark.sh \
  > analysis/5xzr_8j6_benchmark_20260611/logs/launch_second.log 2>&1 &
```

UniDock 对接：

```bash
nohup bash analysis/5xzr_8j6_benchmark_20260611/launch_docking_six_models.sh \
  > analysis/5xzr_8j6_benchmark_20260611/docking/launch_docking.log 2>&1 &
```

## 小规模 smoke 命令

已完成一次 21 molecules / 100 steps smoke，结果归档在：

- `smoke_21_100step_20260611_201840/`
- `smoke_21_100step_20260611_201840/summary/benchmark_summary.csv`
- `smoke_21_100step_20260611_201840/summary/smoke_docking_scores.png`
- `smoke_21_100step_20260611_201840/summary/smoke_quality_summary.png`

先测试核心三模型链路：

```bash
TOTAL=21 TIMESTEPS=100 GPUS="0 1 2" \
  bash analysis/5xzr_8j6_benchmark_20260611/launch_core_benchmark.sh
```

先测试补充三模型链路：

```bash
TOTAL=21 TIMESTEPS=100 GPUS="0 1 2" \
  bash analysis/5xzr_8j6_benchmark_20260611/launch_second_benchmark.sh
```

## 注意

`unidock_tools proteinprep` 和 `mk_prepare_receptor.py` 在当前环境里分别遇到无氢输入和 Meeko/RDKit 兼容问题。本次 receptor PDBQT 使用 OpenBabel 生成：

```bash
obabel -ipdb inputs/5XZR_8J6_pocket8A.pdb -opdbqt \
  -O inputs/5XZR_8J6_pocket8A.pdbqt \
  --partialcharge gasteiger -xr -xh -p 7.4
```

该 PDBQT 已通过 native ligand UniDock sanity check。若后续要做更严格的论文级 docking，可再修复 Meeko/unidock_tools receptor preparation，生成更标准 receptor 后重复 docking 对比。

DiffInt/DiffPharma 的预处理会用 RDKit sanitize 读取参考配体；OpenBabel 从 PDB 转出的原始 8J6 SDF 中一个环内 N 缺少正电荷表示，导致 `Explicit valence for atom # 12 N, 4`。因此本目录额外保留 `5XZR_8J6_ligand_rdkit_fixed.sdf`，只用于 interaction 模型预处理；native UniDock sanity 仍使用原始 ligand SDF。
