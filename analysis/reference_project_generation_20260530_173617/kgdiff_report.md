# KGDiff 单项目安装与生成测试报告

## 结论

- 项目路径：`reference project/pharmacophore/KGDiff`
- 独立环境：`kgdiff_ref`
- checkpoint：已从官方 Zenodo 下载并解压
- 生成测试：成功
- 测试入口：`scripts/sample_for_pocket.py`
- 测试规模：1 个本地 pocket，1 个样本，1000 denoise steps，`joint` guide
- 模型内部采样耗时：`123.50 s`
- 外层 wall time：`real 131.57 s`
- 重建结果：
  - 原子数：22
  - SMILES：`CC1=CCc2ncc(C3CCC4CC=C(O)CN4C3)cc2C1`
- 输出文件：
  - `analysis/reference_project_generation_20260530_173617/kgdiff_outputs/min_pocket/result_rec.pt`
  - `analysis/reference_project_generation_20260530_173617/kgdiff_outputs/min_pocket/result_rec_reconstructed.sdf`
  - `analysis/reference_project_generation_20260530_173617/kgdiff_outputs/min_pocket/sample.yml`
  - `analysis/reference_project_generation_20260530_173617/kgdiff_outputs/min_pocket/log.txt`

## 环境建立过程

KGDiff 官方 README 要求 Python 3.9、PyTorch 1.11、PyG 2.0.4、RDKit、OpenBabel、LMDB、easydict 等。它与 DiffSMol 的运行依赖高度接近，所以为了减少重复下载和降低安装失败概率，我用已有的 `diffsmol_ref` 克隆出独立 conda 环境：

```bash
conda create -n kgdiff_ref --clone diffsmol_ref -y
```

随后在 `kgdiff_ref` 中验证核心依赖：

```text
torch 1.11.0
torch cuda 11.5
torch_geometric 2.0.4
rdkit/lmdb/easydict/yaml import ok
GPU: NVIDIA GeForce RTX 3080
```

注意：普通沙箱下 `torch.cuda.is_available()` 为 `False`，提权 GPU 命令下为 `True`；实际生成测试在 GPU 可见的提权环境中完成。

## checkpoint 下载

官方 README 指向 Zenodo record `8419944`。本次下载的是官方 checkpoint 包：

```text
https://zenodo.org/records/8419944/files/logs_diffusion.zip?download=1
```

下载位置：

```text
analysis/reference_project_generation_20260530_173617/kgdiff_downloads/logs_diffusion.zip
```

大小：

```text
logs_diffusion.zip  139M
kgdiff_assets       159M
```

解压后包含 5 个 checkpoint：

```text
training_2023_06_06__19_20_16/checkpoints/671000.pt
training_2023_06_06__16_05_56/checkpoints/1292000.pt
training_2023_06_06__20_16_16/checkpoints/642000.pt
training_2023_07_05__12_15_55/checkpoints/1076000.pt
training_2023_09_29__14_08_47/checkpoints/1415000.pt
```

其中本次 `joint` 生成实际使用：

```text
analysis/reference_project_generation_20260530_173617/kgdiff_assets/logs_diffusion/training_2023_06_06__20_16_16/checkpoints/642000.pt
```

官方完整数据包 `data.zip` 也在 Zenodo 上，但 HEAD 显示大小约 `4.40 GB`。为了尽快完成生成测试，并避免把大数据集展开到参考项目源码中，本次没有下载完整 `data.zip`；改用 `sample_for_pocket.py` 的自定义 pocket 路线。

## 最小 pocket 输入

为了不改 KGDiff 源码和原始数据目录，我把 BInD 自带的 sample pocket 拷贝到本次 analysis 目录：

```text
analysis/reference_project_generation_20260530_173617/kgdiff_pocket_min/rec.pdb
analysis/reference_project_generation_20260530_173617/kgdiff_pocket_min/ref.sdf
analysis/reference_project_generation_20260530_173617/kgdiff_pocket_min/index.pkl
```

`index.pkl` 只包含一个条目：`rec.pdb` + `ref.sdf`。

## 运行时处理

和 DiffSMol 类似，当前 shell 的外部 `PYTHONPATH` 会污染 KGDiff 的 `utils` 包导入，所以生成命令显式设置：

```text
PYTHONPATH=analysis/.../kgdiff_runtime_patch:reference project/pharmacophore/KGDiff
```

此外 KGDiff 旧代码使用了 NumPy 1.20 后移除的别名：`np.long`、`np.bool`、`np.int`。我没有修改 KGDiff 源码，而是在 analysis 目录下加入运行时补丁：

```text
analysis/reference_project_generation_20260530_173617/kgdiff_runtime_patch/sitecustomize.py
```

该补丁只恢复旧别名到 Python 内置类型。

## 生成命令

工作目录：

```text
reference project/pharmacophore/KGDiff
```

核心命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/kgdiff_runtime_patch:/data/disk6/DiffB110/reference\ project/pharmacophore/KGDiff \
PYTHONDONTWRITEBYTECODE=1 \
timeout 1200 /usr/bin/time -p \
conda run -n kgdiff_ref python scripts/sample_for_pocket.py \
  --config /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/kgdiff_sample_min.yml \
  --pdb_idx 0 \
  --protein_root /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/kgdiff_pocket_min \
  --guide_mode joint \
  --type_grad_weight 100 \
  --pos_grad_weight 25 \
  --batch_size 1 \
  --result_path /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/kgdiff_outputs/min_pocket
```

## 生成结果

程序成功跑完 1000 步，并输出：

```text
Sample done!
real 131.57
user 145.40
sys 5.43
```

从 `result_rec.pt` 读取到模型内部计时：

```text
time_list = [123.49868559837341]
```

使用 KGDiff 自己的 `reconstruct_from_generated()` 从最终步重建分子：

```text
num_atoms = 22
smiles = CC1=CCc2ncc(C3CCC4CC=C(O)CN4C3)cc2C1
```

输出文件大小：

```text
result_rec.pt                  1015428 bytes
result_rec_reconstructed.sdf      1942 bytes
sample.yml                         822 bytes
log.txt                           8449 bytes
```

## 源码状态

KGDiff 子仓库当前 `git status --short` 为空。本次没有修改 KGDiff 源码、官方配置或 checkpoint；新增的配置、运行时补丁、最小 pocket、checkpoint 解压目录和输出都在 `analysis/reference_project_generation_20260530_173617/` 下。

## 对 DiffB110 的直接启示

KGDiff 的优势是显式 knowledge guidance：它在采样中把解释性/亲和相关梯度作为引导项注入，对“原子类型分布更合理”和“口袋相关性更强”有借鉴价值。但速度不理想：单样本 1000 步约 2 分 12 秒外层时间，慢于 DiffSBDD 的快速生成诉求。对你的主线更现实的借鉴方式不是直接替换成 KGDiff，而是吸收它的 guided sampling 思路：把口袋-配体相互作用/类型梯度作为可选 guidance 或 reward-shaping 模块，保留 DiffSBDD 或其它更快底座的采样速度。
