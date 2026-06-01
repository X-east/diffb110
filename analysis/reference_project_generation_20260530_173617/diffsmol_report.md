# DiffSMol 单项目安装与生成测试报告

## 结论

- 项目路径：`reference project/pharmacophore/DiffSMol`
- 独立环境：`diffsmol_ref`
- checkpoint：本地已存在，无需重新下载
  - `models/diffusion.pt`，约 93.5 MB
  - `models/se.pt`，约 3.4 MB
- 生成测试：成功
- 测试规模：CrossDocked processed LMDB，`data_id=0`，`batch_size=1`
- 模型内部采样耗时：`49.16 s`
- 外层 wall time：`real 59.77 s`
- 输出文件：
  - `analysis/reference_project_generation_20260530_173617/diffsmol_outputs/crossdocked_min/result_0.pt`
  - `analysis/reference_project_generation_20260530_173617/diffsmol_outputs/crossdocked_min/sample.yml`

## 环境建立过程

官方 `environment.yml` 的 conda 阶段基本可安装，创建了 `diffsmol_ref` 环境，并安装了 PyTorch 1.11.0 CUDA 11.5、pytorch3d 等核心包；但 pip 阶段失败在固定版本依赖：

```text
autodocktools-py3==1.5.7.post1+12.ga62a5d9
```

该 exact 版本在当前 pip 源无法解析。因此后续没有继续强行改项目源码，而是在独立环境里手动补齐生成测试需要的核心包，包括 PyG 相关 wheel、RDKit、LMDB、easydict、trimesh、scikit-learn、scikit-image、pandas、matplotlib、networkx、openbabel-wheel、oddt、cloudpickle 等。

## checkpoint 与数据

本地已经有 DiffSMol 官方/参考权重：

```text
reference project/pharmacophore/DiffSMol/models/diffusion.pt
reference project/pharmacophore/DiffSMol/models/se.pt
```

本地也已经有可直接采样的 CrossDocked processed LMDB：

```text
reference project/pharmacophore/DiffSMol/data/crossdocked/test/crossdocked_processed_dgcnn_signeddist_512_test.lmdb
```

我没有把 checkpoint 或数据复制进项目源码目录之外的新位置，只把最小采样配置、运行时兼容补丁和生成输出放在本次 analysis 目录下。

## 运行时处理

直接跑 no-pocket MOSES2 路线会触发数据预处理，并遇到 ODDT/scikit-image 的 `marching_cubes` 兼容问题；而且这条路线会尝试在 DiffSMol 原始 `data/MOSES2` 下生成 LMDB。为避免污染参考项目，最终改用已有的 CrossDocked processed LMDB 做最小生成测试。

当前 shell 的 `PYTHONPATH` 中带有外部 `targetdiff` 路径，会污染 DiffSMol 的 `utils` 包导入。因此正式运行时显式设置了独立 `PYTHONPATH`，只加入本次 analysis 下的运行时补丁目录：

```text
analysis/reference_project_generation_20260530_173617/diffsmol_runtime_patch/sitecustomize.py
```

该补丁只做一件事：临时提供 `np.int = int`，解决旧代码与 NumPy 1.24 的兼容问题；没有修改 DiffSMol 源码。

## 生成命令

工作目录：

```text
reference project/pharmacophore/DiffSMol/source
```

核心命令：

```bash
PYTHONPATH=/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/diffsmol_runtime_patch \
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
MPLCONFIGDIR=/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/diffsmol_mplconfig \
timeout 1200 /usr/bin/time -p \
conda run -n diffsmol_ref python -m scripts.sample_diffusion_with_pocket \
  /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/diffsmol_sample_crossdocked_min.yml \
  --data_id 0 \
  --batch_size 1 \
  --result_path /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/diffsmol_outputs/crossdocked_min
```

## 生成结果

程序输出中显示：

```text
time: [49.16498947143555]
Sample done!
real 59.77
user 76.21
sys 10.25
```

结果文件大小：

```text
result_0.pt  3592050 bytes
sample.yml   1161 bytes
```

## 源码状态

DiffSMol 子仓库当前 `git status --short` 为空。本次只新增/使用 analysis 目录下的配置、运行时补丁、matplotlib cache 和输出文件，没有修改 DiffSMol 源码、checkpoint 或原配置。

## 对 DiffB110 的直接启示

DiffSMol 适合作为“相对轻量的药效团/口袋条件扩散参考”，但当前环境复现成本不低：旧依赖、`autodocktools-py3` 固定版本、NumPy 兼容、ODDT/scikit-image 兼容都需要处理。速度上，单口袋单样本 GPU 采样外层约 60 秒，明显慢于 DiffSBDD 的快速采样诉求；它更适合借鉴其 pharmacophore/口袋几何条件表达与评价流程，而不是直接替换 DiffSBDD 作为高速底座。
