# DePPA 单项目安装与生成测试报告

## 结论

- 项目路径：`reference project/DePPA`
- 独立环境：`deppa_ref`
- checkpoint：已从官方 Google Drive 下载
- 生成测试：成功
- 测试入口：`test.py`
- 测试规模：1 个本地 pocket，1 个样本，500 denoise steps
- 模型内部 pocket time：`6.62 s`
- 外层 wall time：`real 18.40 s`
- 生成有效率：`100.00%`（本次 1/1）
- 重建结果：
  - 原子数：10
  - SMILES：`O=C1NC2=C(C=CC=C2)S1`
- 输出文件：
  - `analysis/reference_project_generation_20260530_173617/deppa_outputs/min_test/raw/rec_1_gen.sdf`
  - `analysis/reference_project_generation_20260530_173617/deppa_outputs/min_test/processed/rec_1_gen.sdf`
  - `analysis/reference_project_generation_20260530_173617/deppa_outputs/min_test/pocket_times.txt`

## 环境建立过程

DePPA README 推荐 Python 3.10、Torch 2.1.0 CUDA 11.8、RDKit、OpenBabel、Biopython、torch-scatter、PyTorch Lightning、spyrmsd、qvina、posecheck 等。当前 DiffB110 的 `dif` 环境已经接近 DePPA 生成路径需求：Python 3.10、Torch 2.0.1+cu118、RDKit、OpenBabel、Biopython、torch_scatter、wandb 等都已存在。因此本次先克隆 `dif`，建立独立环境：

```bash
conda create -n deppa_ref --clone dif -y
```

克隆后发现 `deppa_ref/bin/python3.10` 可执行位丢失，导致 `conda run` 报 `Permission denied`。已只对该独立环境修复：

```bash
chmod +x /home/md666/miniforge3/envs/deppa_ref/bin/python3.10
```

随后补齐 DePPA 导入所需的缺包：

```bash
conda install -n deppa_ref -c conda-forge spyrmsd -y
conda run -n deppa_ref python -m pip install gdown
```

核心依赖验证：

```text
torch 2.0.1+cu118
torch_scatter 2.1.2+pt20cu118
pytorch_lightning 1.9.4
rdkit/openbabel/Bio import ok
LigandPocketDDPM import ok
test.py import ok
```

说明：这不是 README 的完全同版环境，主要差异是 Torch 2.0.1 而非 2.1.0、Lightning 1.9.4 而非 2.5.5。但官方 checkpoint 能加载并完成生成。Lightning 在加载时提示从 v1.8.1 自动升级到 v1.9.4；本次没有把升级写回 checkpoint 文件。

## checkpoint 下载

README 给出的官方 checkpoint 链接是 Google Drive file id：

```text
1Ri0Nf4yfW73EntpOE62MrVQOYoSrEk2a
```

下载命令：

```bash
conda run -n deppa_ref gdown 1Ri0Nf4yfW73EntpOE62MrVQOYoSrEk2a \
  -O analysis/reference_project_generation_20260530_173617/deppa_downloads/crossdocked_fullatom_cond.ckpt
```

下载成功，大小约 `17.9 MB`。加载后模型信息：

```text
LigandPocketDDPM
T = 500
mode = pocket_conditioning
pocket_representation = full-atom
```

官方 CrossDocked 数据集也在 Google Drive 文件夹中，但本次为了最小生成测试，没有下载整套数据；改用本地 BInD sample pocket 组织成 DePPA `test.py` 需要的格式。

## 最小 pocket 输入

输入目录：

```text
analysis/reference_project_generation_20260530_173617/deppa_test_pocket/
```

文件：

```text
rec.pdb
rec_1.sdf
rec_1.txt
```

其中 `rec.pdb` 和 `rec_1.sdf` 来自 BInD 的 sample pocket/ligand。`rec_1.txt` 是从 `rec.pdb` 解析出的 residue id 列表，供 `test.py` 指定 pocket residues。

## 生成命令

工作目录：

```text
reference project/DePPA
```

核心命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/disk6/DiffB110/reference\ project/DePPA \
MPLCONFIGDIR=/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/deppa_mplconfig \
PYTHONDONTWRITEBYTECODE=1 \
timeout 1200 /usr/bin/time -p \
conda run -n deppa_ref python test.py \
  /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/deppa_downloads/crossdocked_fullatom_cond.ckpt \
  --test_dir /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/deppa_test_pocket \
  --outdir /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/deppa_outputs/min_test \
  --n_samples 1 \
  --batch_size 1 \
  --timesteps 500
```

## 生成结果

程序输出：

```text
Last processed: rec_1. Validity: 100.00%. 6.62 sec/mol.
Time per pocket: 6.623 +/- 0.00
real 18.40
user 21.35
sys 1.78
```

`pocket_times.txt`：

```text
/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/deppa_test_pocket/rec_1.sdf 6.622676134109497
```

RDKit 读取 processed SDF：

```text
num_mols = 1
num_atoms = 10
smiles = O=C1NC2=C(C=CC=C2)S1
```

输出文件大小：

```text
processed/rec_1_gen.sdf  920 bytes
raw/rec_1_gen.sdf        920 bytes
pocket_times.txt         121 bytes
pocket_times/rec_1.txt   120 bytes
```

## 源码状态

DePPA 子仓库当前 `git status --short` 为空。本次没有修改 DePPA 源码、官方配置或 checkpoint；新增的下载文件、最小输入、输出和 matplotlib cache 都在 `analysis/reference_project_generation_20260530_173617/` 下。

## 对 DiffB110 的直接启示

DePPA 对你现在的问题非常值得借鉴：它不是单纯换一个生成模型，而是在 pocket-aware diffusion 上做 denoising policy optimization，把 QED、SA、vina_score、距离等 reward 组合进 RL/GRPO 式优化路径。它的速度也很好：本次单样本 500 步内部约 `6.62 s`，外层 `18.40 s`，比 KGDiff/BInD/DiffSMol 都轻很多。对你的主线，DePPA 更像是“如何在较快 3D diffusion 上做 RL 优化”的近邻参考，尤其适合借鉴它的按步 denoising policy update、reward 组合和快速 batch 生成测试方式。
