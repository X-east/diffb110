# DiffMeta 独立复现实测报告

## 1. 项目定位

- 项目路径：`reference project/DiffMeta-main`
- 类型：2D graph diffusion / RL 分子生成，不是 3D pocket-conditioned ligand 生成。
- 本次目标：独立环境、官方 pretrained checkpoint、最小分子生成、记录耗时。
- 源码改动：未修改 DiffMeta 源码；所有兼容补丁、最小数据和测试脚本均放在 `analysis/reference_project_generation_20260530_173617/`。

## 2. 环境

- 独立环境：`diffmeta_ref`
- 建立方式：从 `kgdiff_ref` 克隆后安装 DiffMeta 依赖与缺失包。
- 关键版本：

| 项 | 版本 |
| --- | --- |
| Python | 3.9.19 |
| PyTorch | 1.11.0 + CUDA 11.5 |
| PyG | 2.0.4 |
| PyTorch Lightning | 1.7.2 |
| NumPy | 1.24.4 |
| SciPy | 1.9.1 |
| RDKit | 2023.09.2 |

GPU 权限下确认：

```text
cuda_available True
device_count 1
device0 NVIDIA GeForce RTX 3080
```

安装/修复要点：

- `pip install -r requirements.txt` 后曾引入 NumPy 2.x / SciPy 1.13 兼容问题，已固定为 `numpy<2`、`scipy==1.9.1`。
- 额外补充：`vina==1.2.7`、`meeko`、`gemmi`。
- `scorer/ml_predictor_wrapper.py` 在 import 阶段会把 MetaCYP 模型放到 CUDA，因此真实运行必须用 GPU 可见环境。
- `graph_tool` 只被分子路径顶层 import 连带触发，本次 2D molecular sampling 不使用 sbm/planar 图奖励；用 analysis 内 runtime shim 满足 import。
- PyG 2.0.4 缺少 `torch_geometric.typing.InputEdges`，用 analysis 内 `sitecustomize.py` 补类型别名。

## 3. Checkpoint

官方 README 指向 Google Drive pretrained folder：

```text
https://drive.google.com/drive/folders/1J1_r7mf3-SyEqgtl1pv5Bu4r6zmZlbPl
```

已下载：

| 文件 | 字节 |
| --- | ---: |
| `diffmeta_downloads/mosespretrained.pt` | 65,189,961 |
| `diffmeta_downloads/chemblpretrained.pt` | 65,237,639 |

本次生成使用：

```text
analysis/reference_project_generation_20260530_173617/diffmeta_downloads/mosespretrained.pt
```

注意：默认 `configs/model/discrete.yaml` 是 5 层、`dy=64`，与 `mosespretrained.pt` 不匹配；checkpoint 权重对应 `configs/experiment/moses_ppo.yaml` 的 12 层、`dy=128`、`hidden_mlp_dims.y=256` 配置。

## 4. 最小测试输入与运行方式

DiffMeta 官方 `test_only` 分支会硬编码生成 5000 个分子，不能通过 YAML 覆盖为 1 个。为避免一次测试跑得过久，本次新增 analysis 内最小脚本：

```text
analysis/reference_project_generation_20260530_173617/diffmeta_min_generate.py
```

脚本复用 DiffMeta 的 datamodule、MOSES infos、extra features、`DiscreteDenoisingDiffusion`、checkpoint 加载和 `sample_batch()`，只把生成数降为 1。

最小 MOSES 数据：

```text
analysis/reference_project_generation_20260530_173617/diffmeta_min_moses/raw/train_moses.csv
analysis/reference_project_generation_20260530_173617/diffmeta_min_moses/raw/val_moses.csv
analysis/reference_project_generation_20260530_173617/diffmeta_min_moses/raw/test_moses.csv
```

实际生成命令核心：

```bash
/usr/bin/time -p env CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=analysis/reference_project_generation_20260530_173617/diffmeta_runtime_patch:reference\ project/DiffMeta-main:reference\ project/DiffMeta-main/scorer \
  PYTHONDONTWRITEBYTECODE=1 \
  conda run -n diffmeta_ref python analysis/reference_project_generation_20260530_173617/diffmeta_min_generate.py \
  --project-root "reference project/DiffMeta-main" \
  --datadir analysis/reference_project_generation_20260530_173617/diffmeta_min_moses \
  --checkpoint analysis/reference_project_generation_20260530_173617/diffmeta_downloads/mosespretrained.pt \
  --out-dir analysis/reference_project_generation_20260530_173617/diffmeta_outputs/min_moses_direct \
  --num-samples 1
```

## 5. 生成结果与耗时

| 指标 | 值 |
| --- | ---: |
| 生成数量 | 1 |
| diffusion steps | 500 |
| setup/load 时间 | 0.4433 s |
| 纯采样时间 | 18.9253 s |
| 脚本内部总时间 | 19.3686 s |
| `/usr/bin/time real` | 28.35 s |
| `/usr/bin/time user` | 30.95 s |
| `/usr/bin/time sys` | 7.41 s |
| valid_ratio | 1.0 |
| unique_ratio | 0.99999999 |

生成 SMILES：

```text
Cc1ccccc1NC(=O)CCn1cccc1
```

输出文件：

```text
analysis/reference_project_generation_20260530_173617/diffmeta_outputs/min_moses_direct/generated_smiles.csv
analysis/reference_project_generation_20260530_173617/diffmeta_outputs/min_moses_direct/generated_graphs.pt
analysis/reference_project_generation_20260530_173617/diffmeta_outputs/min_moses_direct/timing.json
```

`generated_smiles.csv` 内容：

```text
sample_id,smiles,valid
0,Cc1ccccc1NC(=O)CCn1cccc1,True
```

## 6. 对 DiffB110 的借鉴价值

- DiffMeta 可借鉴的是 2D graph diffusion + RL/property scorer 的思路，不适合作为直接替换 DiffSBDD 的 3D pocket-conditioned 主模型。
- 它的生成速度在 1 个 2D 分子、500 steps 下约 18.9 s 纯采样；对比 3D pocket 项目不完全公平，但说明 12 层 graph transformer 并不算轻。
- 项目工程成熟度一般：官方 test-only 生成数硬编码、环境依赖脆弱、import 阶段耦合 MetaCYP/CUDA/graph_tool。若要借鉴，建议只抽取 scorer/RL 奖励设计，不建议整体并入 DiffB110 主生成链路。
