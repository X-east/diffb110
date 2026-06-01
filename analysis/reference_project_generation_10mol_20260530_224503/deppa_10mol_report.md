# DePPA 10 分子生成测试报告

## 结论

- 项目路径：`reference project/DePPA`
- 独立环境：`deppa_ref`
- checkpoint：`analysis/reference_project_generation_20260530_173617/deppa_downloads/crossdocked_fullatom_cond.ckpt`
- 测试规模：1 个 pocket，10 samples，500 denoise steps，`batch_size=10`
- 生成状态：成功
- processed SDF 可读分子：10/10
- 内部 pocket time：10.6422 s
- 程序显示速度：1.06 s/mol
- 外层 wall time：23.67 s
- 平均外层速度：2.37 s/mol

## 运行命令

```bash
/usr/bin/time -p env CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=reference\ project/DePPA \
  PYTHONDONTWRITEBYTECODE=1 \
  conda run -n deppa_ref python test.py \
  analysis/reference_project_generation_20260530_173617/deppa_downloads/crossdocked_fullatom_cond.ckpt \
  --test_dir analysis/reference_project_generation_20260530_173617/deppa_test_pocket \
  --outdir analysis/reference_project_generation_10mol_20260530_224503/deppa_outputs/min_test_10 \
  --n_samples 10 \
  --batch_size 10 \
  --timesteps 500
```

## 时间

```text
Time per pocket: 10.642 +/- 0.00
Last processed: rec_1. Validity: 100.00%. 1.06 sec/mol.
real 23.67
user 26.90
sys 1.84
```

## 生成 SMILES

```text
0  O=C(O)[C@@H]1C=C(O)C(O)=CC1=O
1  CCC(=O)[C@H]1SC(C(=O)O)=CC1=O
2  C1NCNN1
3  O=CC1OC[C@H](O)[C@H]1Cl
4  CC(=O)CC(O)O
5  CCNC[C@H](C[C@H]1OC1=O)[C@@H]1CN[C@](C)([C@H](O)C[C@H]2CO2)CO1
6  O[C@H]1CCCOC1
7  O=C(O)CC1SS1
8  O=C1OC[C@H](O)COc2ccccc21
9  NO[C@@H]1C=CC(C(=O)O)CC1
```

## 输出文件

```text
analysis/reference_project_generation_10mol_20260530_224503/deppa_outputs/min_test_10/raw/rec_1_gen.sdf
analysis/reference_project_generation_10mol_20260530_224503/deppa_outputs/min_test_10/processed/rec_1_gen.sdf
analysis/reference_project_generation_10mol_20260530_224503/deppa_outputs/min_test_10/pocket_times.txt
```

## 判断

DePPA 的批量吞吐明显优于上一轮单分子估计：内部约 1.06 s/mol，外层约 2.37 s/mol。它仍然是这批 3D pocket-conditioned 项目中速度最友好的候选，值得优先深挖其 denoising policy optimization、reward 组合和采样批处理方式。
