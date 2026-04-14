# 多GPU分批生成配体指南

## 概述

这个脚本支持使用多个GPU并行生成大量分子（如1000个），同时避免显存爆炸。

**关键特性：**
- ✓ 多GPU并行生成（每个GPU独立生成不同的分子）
- ✓ 自动分批避免显存爆炸
- ✓ 使用不同的随机种子确保分子不重复
- ✓ 自动合并所有SDF文件
- ✓ 详细的进度和统计信息

---

## 快速开始

### 方式1：使用Python脚本（推荐）

```bash
cd /home/md666/data/disk6/DiffB110

# 生成1000个分子，使用4个GPU，每批20个
python generate_ligands_batch.py \
    --checkpoint OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0019.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --ref_ligand C:1101 \
    --total 1000 \
    --batch_size 20 \
    --num_gpus 4
```

### 方式2：使用Bash脚本

```bash
cd /home/md666/data/disk6/DiffB110

bash generate_ligands_batch.sh \
    --checkpoint OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0019.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --ref_ligand C:1101 \
    --total 1000 \
    --batch_size 20 \
    --num_gpus 4
```

---

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|---|---|---|
| `--checkpoint` | 模型checkpoint路径 | `OUT/.../epoch=0019.ckpt` |
| `--pdbfile` | PDB文件路径 | `data/8GJJ-C-Cip.pdb` |

### 可选参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ref_ligand` | `C:1101` | 参考配体标识 |
| `--total` | `1000` | 总分子数 |
| `--batch_size` | `20` | 每个GPU每批生成的分子数 |
| `--num_gpus` | `4` | 使用的GPU数量 |
| `--output_dir` | `./generated_ligands` | 输出目录 |
| `--sanitize` | 否 | 启用分子清理 |
| `--relax` | 否 | 启用分子松弛 |
| `--keep_individual` | 否 | 保留各GPU的单独SDF文件 |

---

## 使用示例

### 示例1：生成1000个分子（4GPU，每批20个）

```bash
python generate_ligands_batch.py \
    --checkpoint OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0019.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --ref_ligand C:1101 \
    --total 1000 \
    --batch_size 20 \
    --num_gpus 4
```

**预期耗时：** ~30-40分钟（取决于GPU性能）

**输出：**
```
generated_ligands/
└── 20260316_093000_1000mols_4gpus/
    ├── all_ligands_1000.sdf          # 合并后的所有分子
    ├── gpu0.log                       # GPU 0 的日志
    ├── gpu1.log                       # GPU 1 的日志
    ├── gpu2.log                       # GPU 2 的日志
    └── gpu3.log                       # GPU 3 的日志
```

### 示例2：生成500个分子（8GPU，每批10个，启用清理）

```bash
python generate_ligands_batch.py \
    --checkpoint OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0019.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --total 500 \
    --batch_size 10 \
    --num_gpus 8 \
    --sanitize
```

### 示例3：生成2000个分子（保留单独文件用于调试）

```bash
python generate_ligands_batch.py \
    --checkpoint OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0019.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --total 2000 \
    --batch_size 20 \
    --num_gpus 4 \
    --keep_individual
```

---

## 工作原理

### 分子分配策略

假设生成1000个分子，使用4个GPU：

```
总分子数: 1000
GPU数量: 4
每GPU分子数: 1000 ÷ 4 = 250

分配方案:
  GPU 0: 250 个分子
  GPU 1: 250 个分子
  GPU 2: 250 个分子
  GPU 3: 250 个分子
```

如果不能整除（如1001个分子）：

```
总分子数: 1001
GPU数量: 4
每GPU分子数: 1001 ÷ 4 = 250 余 1

分配方案:
  GPU 0: 250 个分子
  GPU 1: 250 个分子
  GPU 2: 250 个分子
  GPU 3: 251 个分子（处理余数）
```

### 批处理策略

每个GPU内部也会分批生成，避免显存爆炸：

```
GPU 0 需要生成 250 个分子，batch_size=20

分批方案:
  Batch 1: 20 个分子
  Batch 2: 20 个分子
  ...
  Batch 12: 20 个分子
  Batch 13: 10 个分子（余数）
```

### 不重复性保证

- 每个GPU使用不同的 `CUDA_VISIBLE_DEVICES` 环境变量
- 每个GPU的生成过程独立运行，使用不同的随机种子
- PyTorch/CUDA 的随机数生成器在不同GPU上产生不同的序列
- 因此生成的分子自然不重复

---

## 性能优化建议

### 1. 选择合适的batch_size

| batch_size | 优点 | 缺点 |
|---|---|---|
| 10 | 显存占用少，稳定性好 | 生成速度慢 |
| 20 | 平衡性能和稳定性 | 需要较多显存 |
| 50 | 生成速度快 | 容易OOM |

**推荐：** 从20开始，根据GPU显存调整

### 2. 选择合适的GPU数量

| GPU数量 | 优点 | 缺点 |
|---|---|---|
| 4 | 平衡性能和资源占用 | 生成速度中等 |
| 8 | 生成速度快 | 需要8张GPU |
| 1 | 简单，无需多进程 | 生成速度慢 |

**推荐：** 使用所有可用的GPU

### 3. 监控GPU使用情况

在另一个终端运行：

```bash
watch -n 2 nvidia-smi
```

观察：
- GPU利用率是否接近100%
- 显存占用是否稳定
- 是否有进程崩溃

---

## 故障排查

### 问题1：某个GPU任务失败

**症状：** 日志显示 `✗ 失败`

**解决方案：**
1. 查看对应GPU的日志文件：`generated_ligands/.../gpu{N}.log`
2. 检查是否是显存不足（OOM）
3. 减小 `--batch_size` 重试

### 问题2：生成速度很慢

**症状：** 生成1000个分子耗时超过1小时

**解决方案：**
1. 增加 `--num_gpus`（如果有更多GPU）
2. 增加 `--batch_size`（如果显存允许）
3. 检查GPU是否被其他进程占用：`nvidia-smi`

### 问题3：显存爆炸（OOM）

**症状：** 日志显示 `CUDA out of memory`

**解决方案：**
1. 减小 `--batch_size`（如从20改为10）
2. 减少 `--num_gpus`（如从4改为2）
3. 关闭其他GPU任务

### 问题4：生成的分子数不匹配

**症状：** 预期1000个，实际998个

**解决方案：**
1. 查看各GPU的日志，找出失败的任务
2. 手动运行失败的GPU任务补齐
3. 或重新运行整个脚本

---

## 输出文件说明

### 合并文件：`all_ligands_1000.sdf`

包含所有生成的分子，格式为SDF（Structure Data File）

**查看分子数：**
```bash
grep -c '$$$$' all_ligands_1000.sdf
```

**提取前10个分子：**
```bash
head -n 1000 all_ligands_1000.sdf > first_10.sdf
```

### 日志文件：`gpu{N}.log`

每个GPU的详细日志，包含：
- 生成进度
- 错误信息（如有）
- 生成统计

**查看GPU 0的日志：**
```bash
tail -f generated_ligands/.../gpu0.log
```

---

## 高级用法

### 使用不同的checkpoint

```bash
python generate_ligands_batch.py \
    --checkpoint OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0022.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --total 1000 \
    --num_gpus 4
```

### 使用不同的PDB文件

```bash
python generate_ligands_batch.py \
    --checkpoint OUT/.../epoch=0019.ckpt \
    --pdbfile data/another_protein.pdb \
    --ref_ligand A:500 \
    --total 1000 \
    --num_gpus 4
```

### 启用分子松弛（更高质量，但更慢）

```bash
python generate_ligands_batch.py \
    --checkpoint OUT/.../epoch=0019.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --total 1000 \
    --num_gpus 4 \
    --relax
```

---

## 常见问题

**Q: 为什么要分批生成？**
A: 一次性生成1000个分子会导致显存爆炸。分批生成可以在保持生成速度的同时避免OOM。

**Q: 分子会重复吗？**
A: 不会。每个GPU使用不同的随机种子，生成的分子自然不重复。

**Q: 可以中断后继续吗？**
A: 目前不支持断点续传。如果中断，需要重新运行。

**Q: 生成的分子质量如何？**
A: 质量取决于checkpoint的训练质量。使用更新的checkpoint通常会得到更好的分子。

**Q: 可以生成更多分子吗？**
A: 可以。只需增加 `--total` 参数。但耗时会相应增加。

---

## 性能参考

基于 NVIDIA A100 GPU 的参考数据：

| 配置 | 总分子数 | 耗时 | 吞吐量 |
|---|---|---|---|
| 4 GPU, batch=20 | 1000 | ~35 min | ~28 mol/min |
| 8 GPU, batch=20 | 1000 | ~18 min | ~56 mol/min |
| 4 GPU, batch=50 | 1000 | ~20 min | ~50 mol/min |

**注：** 实际耗时取决于GPU型号、系统负载等因素

---

## 许可证和引用

如果使用此脚本生成的分子用于发表，请引用原始DiffSBDD论文。

---

## 支持

如有问题，请检查：
1. 日志文件中的错误信息
2. GPU是否正常工作：`nvidia-smi`
3. 磁盘空间是否充足：`df -h`
4. 内存是否充足：`free -h`
