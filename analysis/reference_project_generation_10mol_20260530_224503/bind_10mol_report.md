# BInD 10 分子批量生成测试报告

## 测试结论

- 项目：`reference project/pharmacophore/BInD`
- 独立环境：`bind_ref`
- checkpoint / public assets：`analysis/reference_project_generation_20260530_173617/bind_assets/public`
- 配置：`analysis/reference_project_generation_10mol_20260530_224503/bind_generate_10.yaml`
- 输出目录：`analysis/reference_project_generation_10mol_20260530_224503/bind_outputs/min_test_10`
- 目标：至少生成 10 个分子，用于比单分子更稳定的速度估计。
- 实际结果：脚本把 `n_generate: 10` 解释为每个测试 pocket 生成 10 个；本次 `Generating with 5 test data`，所以实际落盘 50 个 SDF。
- RDKit sanitize 可读：27 / 50。
- RDKit sanitize 失败：23 / 50，主要是 kekulize 失败和显式价态超限。

## 运行命令

```bash
env CUDA_VISIBLE_DEVICES=0 \
  MPLCONFIGDIR=/home/md666/data/disk6/DiffB110/analysis/reference_project_generation_10mol_20260530_224503/bind_mplconfig \
  PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/time -p conda run -n bind_ref python generate_test_pockets.py \
  /home/md666/data/disk6/DiffB110/analysis/reference_project_generation_10mol_20260530_224503/bind_generate_10.yaml
```

## 速度

| 指标 | 数值 |
|---|---:|
| 生成目标写法 | `n_generate: 10`, `bs: 5` |
| 实际 SDF 文件数 | 50 |
| RDKit 可读 SDF | 27 |
| `/usr/bin/time real` | 634.41 s |
| `/usr/bin/time user` | 531.48 s |
| `/usr/bin/time sys` | 159.40 s |
| 按实际 50 个落盘分子摊销 | 12.69 s/mol |
| 按 27 个 RDKit 可读分子摊销 | 23.50 s/valid mol |

## 前 10 个 RDKit 可读 SMILES

1. `Cc1c([C@H](CO)ON)cccc1[C@H](C)C(=O)C1=C2NC(=O)NCC=C21`
2. `NC(=O)c1ccc(C(=O)F)cc1`
3. `NC(=O)Cc1cccc(F)c1Cl`
4. `O=C(CO)N[C@@H]1C=C(O)C(Cl)=C1`
5. `NCN1CC(OCC(O)(F)F)=C[C@@H]1N`
6. `C[C@@H](N)N1C(=O)C1C[C@@H]1C(=O)CC[C@@H]1F`
7. `NC1=C(C[C@H](CO)OC(=O)O)C=N[C@H]1N`
8. `CCNCC[C@@](O)(CO)[C@@H]1C[C@@H](O)N(C(=O)c2cnc(N)nc2)[C@@H]2O[C@@H]21`
9. `O=C1C2=CC2(CO)CN1CCCCO`
10. `CO.O[C@H]1CCN(O)[C@H](Cc2cc(F)cc(F)c2)C1`

## 质量观察

BInD 这次速度不理想，主要因为完整 1000 step 采样乘上 5 个测试 pocket，每个 pocket 生成 10 个，最终变成 50 个输出。有效性也一般：RDKit 可读率 54%。从用户当前关心的“更合理原子类型分布、键长、成键算法，同时速度不要太慢”来看，BInD 值得借鉴的是它的 pocket 条件生成、POVME prior atom sampler、可选的 steric / bond distance / bond angle / interaction guidance 接口；但直接作为快速主生成器并不占优，除非进一步减少 diffusion steps、只选单个 pocket、或把它的几何/相互作用 guidance 思路移植到更快主模型上。
