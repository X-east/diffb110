# DiffMeta 10 分子生成测试报告

## 结论

- 项目路径：`reference project/DiffMeta-main`
- 独立环境：`diffmeta_ref`
- checkpoint：`analysis/reference_project_generation_20260530_173617/diffmeta_downloads/mosespretrained.pt`
- 测试规模：10 个 2D graph diffusion samples，500 diffusion steps
- 生成状态：完成
- RDKit 有效：8/10
- 纯采样时间：22.6431 s
- 脚本内部总时间：23.1731 s
- 外层 wall time：32.98 s
- 平均外层速度：3.30 s/sample；按有效分子计 4.12 s/valid mol

## 运行方式

继续使用上一轮 analysis 内的兼容方案：runtime patch 只补 `graph_tool` import shim 与 PyG `InputEdges` 类型别名；最小脚本复用 DiffMeta 的 datamodule、模型构建、checkpoint 加载和 `sample_batch()`，避免官方 `test_only` 分支硬编码生成 5000 个分子。

核心命令：

```bash
/usr/bin/time -p env CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=analysis/reference_project_generation_10mol_20260530_224503/diffmeta_runtime_patch:reference\ project/DiffMeta-main:reference\ project/DiffMeta-main/scorer \
  PYTHONDONTWRITEBYTECODE=1 \
  conda run -n diffmeta_ref python analysis/reference_project_generation_10mol_20260530_224503/diffmeta_min_generate.py \
  --project-root "reference project/DiffMeta-main" \
  --datadir analysis/reference_project_generation_10mol_20260530_224503/diffmeta_min_moses \
  --checkpoint analysis/reference_project_generation_20260530_173617/diffmeta_downloads/mosespretrained.pt \
  --out-dir analysis/reference_project_generation_10mol_20260530_224503/diffmeta_outputs/min_moses_10 \
  --num-samples 10
```

## 时间

```text
setup_load_seconds = 0.5300172900
sample_seconds     = 22.6430800610
total_seconds      = 23.1730973510
real               = 32.98
user               = 37.00
sys                = 7.58
```

## 生成结果

有效 SMILES：

```text
OCCSc1nccn1-c1ccc(Cl)c(Cl)c1
CCOc1ccc(CN2CCOCC2C)cc1OCC
O=Cc1ccccc1NC(=O)c1cccnc1OCC(F)F
O=C(NCCOc1ccc(Cl)cn1)Nc1ccccc1
O=c1c2c(F)cccc2ncn1CC1(c2cccc(F)c2)CC1
Cc1cc(Br)cc(NCc2cnc(C)nc2)c1
Nc1ccnc2nnc(C(=O)Nc3ccccc3)cc12
C=CCN(CC1CCCO1)c1ccc2cc1NC(=O)CS2
```

无效样本：2 个，RDKit 报 `Can't kekulize mol`。

输出文件：

```text
analysis/reference_project_generation_10mol_20260530_224503/diffmeta_outputs/min_moses_10/generated_smiles.csv
analysis/reference_project_generation_10mol_20260530_224503/diffmeta_outputs/min_moses_10/generated_graphs.pt
analysis/reference_project_generation_10mol_20260530_224503/diffmeta_outputs/min_moses_10/timing.json
```

## 判断

DiffMeta 批量生成速度比单分子更合理：10 samples 纯采样 22.64 秒，约 2.26 s/sample；但它是 2D graph diffusion，不提供 pocket-conditioned 3D 坐标和键长分布。对 DiffB110 更适合借鉴 RL/scorer 和 2D property reward，不适合作为 3D SBDD 主模型。
