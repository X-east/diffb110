# DiffGui 生成测试报告

## 项目与环境

- 项目路径：`reference project/DiffGui`
- 独立环境：`diffgui`
- 环境检查：`conda run -n diffgui python -c "import sys, torch, rdkit, torch_geometric, torch_scatter; ..."` 通过
- Python：3.7.7
- PyTorch：1.13.1，CUDA build：11.6
- 当前非提权运行下 `torch.cuda.is_available()` 为 `False`，`torch.cuda.device_count()` 为 `0`

## Checkpoint 与输入

- 主模型 checkpoint：`reference project/DiffGui/ckpt/trained.pt`，102700484 bytes
- bond predictor checkpoint：`reference project/DiffGui/ckpt/bond_trained.pt`，85941431 bytes
- 测试 pocket：`reference project/DiffGui/sample/3ztx_pocket.pdb`
- 本轮未重新下载 checkpoint，因为本地 checkpoint 已存在且大小非零。

## 测试配置

- 最小测试配置：`analysis/reference_project_generation_20260530_173617/diffgui_sample_min.yml`
- 与原始配置相比只在分析目录副本中缩小：
  - `sample.batch_size: 1`
  - `sample.num_mols: 1`
  - `sample.save_traj_prob: 0.0`
- 原项目配置 `reference project/DiffGui/configs/sample/sample.yml` 未修改。

## 运行命令

```bash
timeout 900 /usr/bin/time -p conda run -n diffgui python scripts/sample.py \
  --config /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/diffgui_sample_min.yml \
  --outdir /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/diffgui_outputs \
  --device cpu \
  --batch_size 1
```

## CPU 预跑记录

- 运行开始：2026-05-30 17:38:50
- 运行耗时上限：900 秒
- 退出码：124（`timeout` 终止）
- 生成分子数：0
- 生成输出目录：`analysis/reference_project_generation_20260530_173617/diffgui_outputs/diffgui_sample_min_20260530_173850`
- 判断：普通沙箱下 GPU 不可见，CPU 900 秒仍停在 1 个 ligand 的 1000-step 采样阶段，不能作为有效速度基准。

## GPU 生成结果

GPU 权限下补跑同一份最小配置：

```bash
/usr/bin/time -p env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  conda run -n diffgui python scripts/sample.py \
  --config /home/md666/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/diffgui_sample_min.yml \
  --outdir /home/md666/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/diffgui_outputs_gpu \
  --device cuda:0 \
  --batch_size 1
```

| 指标 | 值 |
| --- | ---: |
| 退出码 | 0 |
| 生成分子数 | 1 |
| ligand atom 数 | 19 |
| diffusion steps | 1000 |
| `/usr/bin/time real` | 212.12 s |
| `/usr/bin/time user` | 208.80 s |
| `/usr/bin/time sys` | 4.31 s |

生成 SMILES：

```text
O=c1c2ccccc2oc2cccc3nccc1c32
```

打分记录：

| sa | qed | vina | score |
| ---: | ---: | ---: | ---: |
| 0.88 | 0.4773784226 | -7.439 | -8.7963784226 |

输出文件：

```text
analysis/reference_project_generation_20260530_173617/diffgui_outputs_gpu/diffgui_sample_min_20260530_223501/SMILES.txt
analysis/reference_project_generation_20260530_173617/diffgui_outputs_gpu/diffgui_sample_min_20260530_223501/3ztx_SDF/0.sdf
analysis/reference_project_generation_20260530_173617/diffgui_outputs_gpu/diffgui_sample_min_20260530_223501/samples_3ztx.pt
analysis/reference_project_generation_20260530_173617/diffgui_outputs_gpu/diffgui_sample_min_20260530_223501/log.txt
```

## 判断

DiffGui 的环境、checkpoint、bond predictor、pocket 输入和 Vina score-only 链路均可运行。真实 GPU 速度约 212 秒/分子（1000 steps、1 个 19 原子 ligand、带 bond predictor + uncertainty guidance + Vina score-only），明显慢于 DePPA/BInD/DiffSMol/KGDiff 的最小测试；但它的优势是显式 bond predictor、生成后带 SDF 和 score-only 评估，适合借鉴成键/重建与 property guidance 思路，不适合作为追求高吞吐的直接主模型。
