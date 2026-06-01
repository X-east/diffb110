# TargetDiff 10 samples 生成测试报告

## 结论

- 项目路径：`/home/md666/data/disk6/XDC/targetdiff`
- 独立环境：`targetdiff`
- checkpoint：`/home/md666/data/disk6/XDC/targetdiff/pretrained_models/pretrained_diffusion.pt`
- 测试输入：8GJL protein + ciprofloxacin ligand 提取 10A pocket
- 测试规模：10 model samples，500 denoise steps，`batch_size=5`
- 生成状态：程序完成
- SDF 写出：1/10
- reconstruction failed：9/10
- 外层 wall time：192.88 s
- 平均外层速度：19.29 s/model sample；按 SDF 有效分子计 192.88 s/valid SDF

## 环境修复

本机已有 `targetdiff` conda 环境，但 `bin/python3.8` 缺少可执行位，首次运行失败：

```text
/home/md666/miniforge3/envs/targetdiff/bin/python: Permission denied
```

已只修复该独立环境：

```bash
chmod +x /home/md666/miniforge3/envs/targetdiff/bin/python3.8 /home/md666/miniforge3/envs/targetdiff/bin/python3.8-config
```

## 运行配置

配置文件：

```text
analysis/reference_project_generation_10mol_20260530_224503/targetdiff_sample_10.yml
```

核心参数：

```yaml
runtime:
  device: cuda:0
  batch_size: 5
sample:
  seed: 2026
  num_samples: 10
  num_steps: 500
  sample_num_atoms: prior
```

## 运行命令

```bash
/usr/bin/time -p env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  conda run -n targetdiff python scripts/sample_from_protein_ligand.py \
  analysis/reference_project_generation_10mol_20260530_224503/targetdiff_sample_10.yml
```

## 时间与日志结果

```text
Finished generation. total_generated=10, sdf_written=1, reconstruction_failed=9
real 192.88
user 775.47
sys 9.74
```

RDKit/OpenBabel 重建阶段出现多次价态问题，例如：

```text
Explicit valence for atom Cl/F is greater than permitted
```

## 生成结果

`generated_smiles.tsv` 只写出 1 个有效条目：

```text
global_index  batch_index  sample_index  seed  smiles
1             0            1             2026  C.C1OCP1.CPC.F.NCCP(F)OSN.NF.O.OC(F)CC(F)CF
```

RDKit 读取 SDF：

```text
num_mols = 1
num_atoms = 28
smiles = C.C1OCP1.CPC.F.NCCP(F)OSN.NF.O.O[C@H](F)CC(F)CF
```

输出文件：

```text
analysis/reference_project_generation_10mol_20260530_224503/targetdiff_outputs/8gjl_cip_10/generated_molecules.sdf
analysis/reference_project_generation_10mol_20260530_224503/targetdiff_outputs/8gjl_cip_10/generated_smiles.tsv
analysis/reference_project_generation_10mol_20260530_224503/targetdiff_outputs/8gjl_cip_10/input_pocket.pdb
analysis/reference_project_generation_10mol_20260530_224503/targetdiff_outputs/8gjl_cip_10/run_config.yml
```

## 判断

TargetDiff 可以跑通 10 samples，但在这个 8GJL/cip pocket 上成键重建质量很差：10 个模型样本只有 1 个写出 SDF，且这个分子是多片段 SMILES。对你的目标而言，这说明原版 TargetDiff 不能直接作为“合理成键/合理键长”的最终解法；它更适合作为 TargetDiff 架构和条件扩散参考，而成键重建需要额外的 bond predictor、价态约束、片段过滤或像 DiffGui/DePPA 那样的后处理与奖励约束。
