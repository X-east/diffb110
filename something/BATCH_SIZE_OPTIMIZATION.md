# Batch Size 优化指南

## 问题
采样batch不能整除训练batch会导致效率低下。

## 解决方案
调整 `sample_group_size` 和 `each_prompt_sample` 使采样batch能整除训练batch。

## 6GPU配置对比

### 原配置（不优化）
```yaml
dataloader:
  sample_group_size: 3
  each_prompt_sample: 24

train:
  batch_size: 192
```

**计算：**
- 采样batch = 3 × 24 = 72
- 192 ÷ 72 = 2.67 ❌ 不能整除，浪费

**问题：**
- 每3个step的采样结果(3×72=216)才能组成1个训练batch(192)
- 浪费24个分子的计算

---

### 优化配置1（推荐）
```yaml
dataloader:
  sample_group_size: 4
  each_prompt_sample: 16

train:
  batch_size: 192
```

**计算：**
- 采样batch = 4 × 16 = 64
- 192 ÷ 64 = 3 ✅ 完美整除

**优势：**
- 每3个step的采样结果(3×64=192)恰好组成1个训练batch
- 零浪费，高效！
- 配置文件：`ddp_6gpu_optimized.yaml`

---

### 优化配置2（更多采样）
```yaml
dataloader:
  sample_group_size: 6
  each_prompt_sample: 16

train:
  batch_size: 192
```

**计算：**
- 采样batch = 6 × 16 = 96
- 192 ÷ 96 = 2 ✅ 完美整除

**优势：**
- 每2个step的采样结果(2×96=192)恰好组成1个训练batch
- 采样更多分子(96 vs 64)，可能获得更好的分子
- 但计算量更大

---

### 优化配置3（单step训练）
```yaml
dataloader:
  sample_group_size: 12
  each_prompt_sample: 16

train:
  batch_size: 192
```

**计算：**
- 采样batch = 12 × 16 = 192
- 192 ÷ 192 = 1 ✅ 完美整除

**优势：**
- 每1个step的采样结果恰好组成1个训练batch
- 最简单，无需梯度累积
- 但采样最多(192个)，计算量最大

---

## 推荐方案

**推荐使用优化配置1：**
```
sample_group_size: 4
each_prompt_sample: 16
采样batch: 64
```

**原因：**
1. ✅ 采样batch能完美整除训练batch
2. ✅ 采样数量适中(64个)，计算效率高
3. ✅ 每3个step组成1个训练batch，梯度累积合理
4. ✅ 平衡了采样多样性和计算效率

---

## 其他GPU数量的优化

### 4GPU配置
```yaml
train:
  batch_size: 64  # 4 × 16

# 方案1：采样batch=64
dataloader:
  sample_group_size: 4
  each_prompt_sample: 16
# 64 ÷ 64 = 1 ✓

# 方案2：采样batch=32
dataloader:
  sample_group_size: 2
  each_prompt_sample: 16
# 64 ÷ 32 = 2 ✓
```

### 8GPU配置
```yaml
train:
  batch_size: 256  # 8 × 32

# 方案1：采样batch=128
dataloader:
  sample_group_size: 8
  each_prompt_sample: 16
# 256 ÷ 128 = 2 ✓

# 方案2：采样batch=64
dataloader:
  sample_group_size: 4
  each_prompt_sample: 16
# 256 ÷ 64 = 4 ✓
```

---

## 如何选择

| 优先级 | 选择 | 配置 |
|------|------|------|
| **效率优先** | 采样batch=64 | sample_group_size=4, each_prompt_sample=16 |
| **多样性优先** | 采样batch=96 | sample_group_size=6, each_prompt_sample=16 |
| **简单性优先** | 采样batch=192 | sample_group_size=12, each_prompt_sample=16 |

---

## 运行优化配置

```bash
cd /home/md666/data/disk6/DiffB110

# 使用优化配置
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
torchrun --nproc_per_node=6 \
  ddpo/main_diffsbdd_lightning.py \
  --config_path ddpo/configs/ddp_6gpu_optimized.yaml \
  --seed 42 \
  --log_level INFO
```

或修改 `run_production_training.sh`：
```bash
CONFIG_PATH="ddpo/configs/ddp_6gpu_optimized.yaml"
```

---

## 性能对比

| 配置 | 采样batch | 整除性 | 采样数 | 计算量 | 推荐度 |
|------|---------|-------|-------|-------|-------|
| 原配置 | 72 | ❌ 2.67 | 72 | 低 | ⭐ |
| 优化1 | 64 | ✅ 3 | 64 | 中 | ⭐⭐⭐⭐⭐ |
| 优化2 | 96 | ✅ 2 | 96 | 高 | ⭐⭐⭐⭐ |
| 优化3 | 192 | ✅ 1 | 192 | 最高 | ⭐⭐⭐ |

