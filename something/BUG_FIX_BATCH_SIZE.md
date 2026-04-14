# Bug修复：batch_size整除问题

## 问题描述

**症状：** 所有GPU任务都失败，错误信息：
```
AssertionError
  File "/data/disk6/DiffB110/DiffSBDD/generate_ligands.py", line 36, in <module>
    assert args.n_samples % args.batch_size == 0
```

**根本原因：** `generate_ligands.py` 第36行有一个断言检查，要求 `n_samples` 必须能被 `batch_size` 整除。

**触发条件：**
- 总分子数：1000
- GPU数量：4
- 每GPU分子数：1000 ÷ 4 = 250
- batch_size：20
- 250 ÷ 20 = 12.5 ❌ **不能整除！**

---

## 修复方案

### 修复1：Bash脚本 (`generate_ligands_batch.sh`)

在计算分配策略后，自动调整batch_size使其能整除每GPU分子数：

```bash
# 调整batch_size使其能整除每GPU分子数
if [ $((SAMPLES_PER_GPU % BATCH_SIZE)) -ne 0 ]; then
    # 找到最大的能整除SAMPLES_PER_GPU的batch_size
    for bs in $(seq $BATCH_SIZE -1 1); do
        if [ $((SAMPLES_PER_GPU % bs)) -eq 0 ]; then
            BATCH_SIZE=$bs
            break
        fi
    done
fi
```

**工作原理：**
- 检查 `SAMPLES_PER_GPU % BATCH_SIZE` 是否为0
- 如果不为0，从 `BATCH_SIZE` 开始向下遍历
- 找到第一个能整除 `SAMPLES_PER_GPU` 的值
- 使用这个新的batch_size

**示例：**
```
SAMPLES_PER_GPU = 250
原始 BATCH_SIZE = 20
250 % 20 = 10 ❌ 不能整除

自动调整：
250 % 19 = 13 ❌
250 % 18 = 16 ❌
250 % 17 = 11 ❌
250 % 16 = 10 ❌
250 % 15 = 10 ❌
250 % 14 = 12 ❌
250 % 13 = 3 ❌
250 % 12 = 10 ❌
250 % 11 = 8 ❌
250 % 10 = 0 ✅ 找到！

新的 BATCH_SIZE = 10
```

### 修复2：Python脚本 (`generate_ligands_batch.py`)

在main函数中添加相同的检查和调整逻辑：

```python
# 确保每个GPU分配的分子数能被batch_size整除
if samples_per_gpu % args.batch_size != 0:
    # 找到最大的能整除samples_per_gpu的batch_size
    original_batch_size = args.batch_size
    for bs in range(args.batch_size, 0, -1):
        if samples_per_gpu % bs == 0:
            args.batch_size = bs
            break
    logger.warning(
        f"每GPU分子数 ({samples_per_gpu}) 不能被batch_size ({original_batch_size}) 整除。"
        f"自动调整batch_size为 {args.batch_size}。"
    )
```

---

## 修复后的行为

### 修复前
```
总分子数: 1000
GPU数量: 4
每GPU分子数: 250
batch_size: 20
250 % 20 = 10 ❌

结果: AssertionError - 所有GPU任务失败
```

### 修复后
```
总分子数: 1000
GPU数量: 4
每GPU分子数: 250
原始batch_size: 20
250 % 20 = 10 ❌

自动调整: batch_size = 10
250 % 10 = 0 ✅

结果: 成功生成1000个分子
```

---

## 测试验证

### 测试用例1：原始失败场景
```bash
python generate_ligands_batch.py --total 1000 --num_gpus 4 --batch_size 20
```

**修复前：** ❌ 所有GPU任务失败
**修复后：** ✅ 自动调整batch_size为10，成功生成

### 测试用例2：已能整除的场景
```bash
python generate_ligands_batch.py --total 1000 --num_gpus 4 --batch_size 10
```

**修复前：** ✅ 成功
**修复后：** ✅ 成功（无需调整）

### 测试用例3：其他不能整除的场景
```bash
python generate_ligands_batch.py --total 500 --num_gpus 3 --batch_size 20
```

**分析：**
- 每GPU分子数：500 ÷ 3 = 166 (余2)
- GPU 0,1: 166个分子
- GPU 2: 168个分子
- 166 % 20 = 6 ❌ 不能整除
- 自动调整：batch_size = 2（166 % 2 = 0 ✅）

---

## 性能影响

修复后batch_size可能会被自动调整为更小的值，这会影响生成速度：

| 场景 | 原始batch_size | 调整后batch_size | 性能影响 |
|---|---|---|---|
| 1000分子 × 4GPU | 20 | 10 | -50% 速度 |
| 1000分子 × 8GPU | 20 | 20 | 无影响 |
| 500分子 × 3GPU | 20 | 2 | -90% 速度 |

**建议：** 选择能被GPU分子数整除的batch_size和GPU数量组合，避免自动调整。

---

## 最佳实践

### 选择合适的参数组合

为了避免自动调整，选择能整除的组合：

| 总分子数 | GPU数量 | 每GPU分子数 | 推荐batch_size |
|---|---|---|---|
| 1000 | 4 | 250 | 10, 25, 50, 125, 250 |
| 1000 | 8 | 125 | 5, 25, 125 |
| 1000 | 5 | 200 | 10, 20, 40, 50, 100, 200 |
| 500 | 4 | 125 | 5, 25, 125 |

### 推荐命令

```bash
# 1000分子 × 4GPU（推荐）
python generate_ligands_batch.py --total 1000 --num_gpus 4 --batch_size 10

# 1000分子 × 8GPU（推荐）
python generate_ligands_batch.py --total 1000 --num_gpus 8 --batch_size 20

# 500分子 × 4GPU（推荐）
python generate_ligands_batch.py --total 500 --num_gpus 4 --batch_size 25
```

---

## 修复验证

### 验证修复已应用

检查脚本是否包含修复代码：

```bash
# 检查Bash脚本
grep -n "SAMPLES_PER_GPU % BATCH_SIZE" generate_ligands_batch.sh

# 检查Python脚本
grep -n "samples_per_gpu % args.batch_size" generate_ligands_batch.py
```

### 运行测试

```bash
# 测试原始失败场景
python generate_ligands_batch.py --total 1000 --num_gpus 4 --batch_size 20

# 应该看到警告信息：
# WARNING - 每GPU分子数 (250) 不能被batch_size (20) 整除。
# 自动调整batch_size为 10。
```

---

## 总结

| 项目 | 详情 |
|---|---|
| **问题** | batch_size不能整除每GPU分子数导致AssertionError |
| **根本原因** | `generate_ligands.py` 的断言检查 |
| **修复方案** | 自动调整batch_size使其能整除 |
| **影响范围** | 两个脚本都已修复 |
| **性能影响** | 可能降低生成速度（取决于参数组合） |
| **建议** | 选择能整除的参数组合避免自动调整 |

---

**修复日期：** 2026-03-16  
**修复版本：** 1.1  
**状态：** ✅ 已验证
