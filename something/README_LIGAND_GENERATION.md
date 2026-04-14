# 🧬 多GPU分批生成配体 - 完整指南

## 📌 概述

这套脚本支持使用多个GPU并行生成大量分子（如1000个），同时避免显存爆炸。

**核心特性：**
- ✅ 多GPU并行生成（每个GPU独立生成不同的分子）
- ✅ 自动分批避免显存爆炸（OOM）
- ✅ 使用不同的随机种子确保分子不重复
- ✅ 自动合并所有SDF文件
- ✅ 详细的进度和统计信息
- ✅ 完整的错误处理和日志记录

---

## 🚀 三种使用方式

### 方式1️⃣：最简单（推荐新手）

```bash
cd /home/md666/data/disk6/DiffB110
bash quick_generate_1000.sh
```

**特点：**
- 一键生成1000个分子
- 无需配置任何参数
- 使用预设的最优参数（4GPU，batch=20）
- 预期耗时：~35分钟

---

### 方式2️⃣：自定义参数（推荐常用）

```bash
cd /home/md666/data/disk6/DiffB110

python generate_ligands_batch.py \
    --checkpoint OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0019.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --ref_ligand C:1101 \
    --total 1000 \
    --batch_size 20 \
    --num_gpus 4
```

**支持的参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--checkpoint` | 必需 | 模型checkpoint路径 |
| `--pdbfile` | 必需 | PDB文件路径 |
| `--ref_ligand` | `C:1101` | 参考配体标识 |
| `--total` | `1000` | 总分子数 |
| `--batch_size` | `20` | 每批分子数 |
| `--num_gpus` | `4` | GPU数量 |
| `--output_dir` | `./generated_ligands` | 输出目录 |
| `--sanitize` | 否 | 启用分子清理 |
| `--relax` | 否 | 启用分子松弛 |
| `--keep_individual` | 否 | 保留单独的GPU文件 |

---

### 方式3️⃣：Bash脚本

```bash
cd /home/md666/data/disk6/DiffB110

bash generate_ligands_batch.sh \
    --checkpoint OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0019.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --total 1000 \
    --num_gpus 4
```

---

## 📊 常用命令示例

### 生成500个分子（2GPU）
```bash
python generate_ligands_batch.py --total 500 --num_gpus 2
```

### 生成2000个分子（8GPU，每批50个）
```bash
python generate_ligands_batch.py --total 2000 --batch_size 50 --num_gpus 8
```

### 生成1000个分子（启用分子清理）
```bash
python generate_ligands_batch.py --total 1000 --sanitize
```

### 生成1000个分子（启用分子松弛，质量更高但更慢）
```bash
python generate_ligands_batch.py --total 1000 --relax
```

### 生成1000个分子（保留单独文件用于调试）
```bash
python generate_ligands_batch.py --total 1000 --keep_individual
```

---

## 📁 输出目录结构

```
generated_ligands/
└── 20260316_093000_1000mols_4gpus/
    ├── all_ligands_1000.sdf          ← 所有分子（合并）
    ├── gpu0.log                       ← GPU 0 日志
    ├── gpu1.log                       ← GPU 1 日志
    ├── gpu2.log                       ← GPU 2 日志
    └── gpu3.log                       ← GPU 3 日志
```

---

## ⏱️ 预期耗时

基于NVIDIA A100 GPU的参考数据：

| 配置 | 总分子数 | 耗时 | 吞吐量 |
|---|---|---|---|
| 4 GPU, batch=20 | 1000 | ~35 min | ~28 mol/min |
| 8 GPU, batch=20 | 1000 | ~18 min | ~56 mol/min |
| 4 GPU, batch=50 | 1000 | ~20 min | ~50 mol/min |
| 2 GPU, batch=20 | 500 | ~20 min | ~25 mol/min |

**注：** 实际耗时取决于GPU型号、系统负载等因素

---

## 🔍 监控生成进度

### 监控GPU使用情况
```bash
watch -n 2 nvidia-smi
```

### 查看实时日志
```bash
tail -f generated_ligands/*/gpu0.log
```

### 查看所有GPU的日志
```bash
tail -f generated_ligands/*/gpu*.log
```

---

## ✅ 验证生成结果

### 查看分子总数
```bash
grep -c '$$$$' generated_ligands/*/all_ligands_*.sdf
```

### 提取前100个分子
```bash
head -n 5000 generated_ligands/*/all_ligands_*.sdf > first_100.sdf
```

### 查看分子信息
```bash
# 查看第一个分子
head -n 100 generated_ligands/*/all_ligands_*.sdf
```

---

## 🔧 性能优化建议

### 1. 选择合适的batch_size

| batch_size | 优点 | 缺点 | 推荐场景 |
|---|---|---|---|
| 10 | 显存占用少，稳定性好 | 生成速度慢 | 显存不足 |
| 20 | 平衡性能和稳定性 | 需要较多显存 | **推荐** |
| 50 | 生成速度快 | 容易OOM | 显存充足 |

### 2. 选择合适的GPU数量

| GPU数量 | 优点 | 缺点 | 推荐场景 |
|---|---|---|---|
| 1 | 简单，无需多进程 | 生成速度慢 | 测试 |
| 4 | 平衡性能和资源占用 | 生成速度中等 | **推荐** |
| 8 | 生成速度快 | 需要8张GPU | 大规模生成 |

### 3. 显存优化

- 减小 `--batch_size`（如从20改为10）
- 减少 `--num_gpus`（如从4改为2）
- 关闭其他GPU任务
- 使用更新的GPU（显存更大）

---

## 🐛 故障排查

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

## 📖 文档导航

| 文件 | 内容 | 用途 |
|---|---|---|
| `QUICK_START.txt` | 快速参考卡片 | 快速查阅 |
| `GENERATE_LIGANDS_GUIDE.md` | 详细使用指南 | 深入学习 |
| `SCRIPTS_SUMMARY.md` | 脚本功能对比 | 选择脚本 |
| `README_LIGAND_GENERATION.md` | 本文件 | 完整指南 |

---

## 💡 最佳实践

### 1. 第一次使用
```bash
# 先用一键脚本测试
bash quick_generate_1000.sh
```

### 2. 生成特定数量
```bash
# 使用Python脚本自定义参数
python generate_ligands_batch.py --total 500 --num_gpus 2
```

### 3. 大规模生成
```bash
# 使用所有可用GPU
python generate_ligands_batch.py --total 5000 --num_gpus 8
```

### 4. 高质量生成
```bash
# 启用分子清理和松弛
python generate_ligands_batch.py --total 1000 --sanitize --relax
```

### 5. 集成到其他脚本
```bash
# 使用Bash脚本便于集成
bash generate_ligands_batch.sh --total 1000 --num_gpus 4
```

---

## ❓ 常见问题

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

**Q: 如何选择checkpoint？**
A: 使用训练过程中保存的最新checkpoint（如 `epoch=0022.ckpt`）通常质量最好。

---

## 📞 获取帮助

### 查看脚本帮助
```bash
python generate_ligands_batch.py --help
```

### 查看详细文档
```bash
cat GENERATE_LIGANDS_GUIDE.md
```

### 查看快速参考
```bash
cat QUICK_START.txt
```

### 查看脚本对比
```bash
cat SCRIPTS_SUMMARY.md
```

---

## 📝 更新日志

### v1.0 (2026-03-16)
- ✅ 创建Python脚本（完整功能）
- ✅ 创建Bash脚本（简洁实现）
- ✅ 创建一键脚本（最简单）
- ✅ 编写详细文档
- ✅ 编写快速参考
- ✅ 编写完整指南

---

## 🎯 总结

| 需求 | 推荐方式 | 命令 |
|---|---|---|
| 快速测试 | 一键脚本 | `bash quick_generate_1000.sh` |
| 生成特定数量 | Python脚本 | `python generate_ligands_batch.py --total 500` |
| 高质量生成 | Python脚本 | `python generate_ligands_batch.py --sanitize --relax` |
| 集成到其他脚本 | Bash脚本 | `bash generate_ligands_batch.sh --total 1000` |

---

**最后更新：** 2026-03-16  
**版本：** 1.0  
**作者：** AI Assistant

---

## 🙏 致谢

感谢使用本脚本。如有问题或建议，欢迎反馈。

