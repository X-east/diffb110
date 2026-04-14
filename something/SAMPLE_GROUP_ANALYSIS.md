# sample_group_size 和 each_prompt_sample 深度分析

## 核心概念

这两个参数控制的是**分子生成的分组采样策略**，而不是简单的数量参数。

### 代码关键片段

```python
# 从 DiffSBDDDataLoader.generate_batch()

# 1. 为每个group采样一个原子数
nodesxsample = torch.randint(
    self.min_nodes, self.max_nodes + 1,
    (self.group_sample,),  # ← sample_group_size
    device=self.device
)
# 结果: [n1, n2, n3, n4] (4个不同的原子数)

# 2. 创建group索引
group_index = torch.arange(self.group_sample, device=self.device)
# 结果: [0, 1, 2, 3]

# 3. 为每个group的原子数复制each_prompt_sample次
group_index = group_index.repeat_interleave(self.each_prompt_sample)
# 结果: [0,0,...,0(16次), 1,1,...,1(16次), ...]

nodesxsample = nodesxsample.repeat_interleave(self.each_prompt_sample)
# 结果: [n1,n1,...,n1(16次), n2,n2,...,n2(16次), ...]
```

---

## 参数含义

### sample_group_size（采样组数）

**定义：** 每个batch中有多少个**不同的原子数条件**

**作用：** 
- 控制分子多样性的**原子数维度**
- 每个group代表一个"原子数条件"
- 模型为每个条件生成多个分子

**例子（sample_group_size=4）：**
```
Group 0: 原子数 = 12 → 生成16个12原子的分子
Group 1: 原子数 = 18 → 生成16个18原子的分子
Group 2: 原子数 = 22 → 生成16个22原子的分子
Group 3: 原子数 = 28 → 生成16个28原子的分子

总共: 4 × 16 = 64个分子
```

### each_prompt_sample（每个提示的采样数）

**定义：** 对于**每个原子数条件**，生成多少个分子

**作用：**
- 控制分子多样性的**采样数维度**
- 同一原子数条件下生成多个不同的分子
- 用于奖励评估和策略更新

**例子（each_prompt_sample=16）：**
```
对于原子数=12的条件，生成16个不同的分子：
- 分子1: C-C-O-N-... (12原子)
- 分子2: C-N-S-O-... (12原子)
- ...
- 分子16: N-O-C-C-... (12原子)
```

---

## 为什么需要两个参数？

### 1. 多维度多样性

单一参数无法同时控制两个维度：

```
只有batch_size=64：
❌ 不知道是 64个相同原子数的分子
❌ 还是 4个不同原子数各16个分子
❌ 还是 8个不同原子数各8个分子

有两个参数：
✅ 明确指定：4个不同原子数，每个16个分子
```

### 2. 奖励评估的需求

DDPO训练需要：
- **多个样本**来估计同一条件下的奖励分布
- **多个条件**来学习条件依赖的策略

```
each_prompt_sample=16 → 同一条件下有16个样本
                        可以计算该条件的平均奖励、方差等

sample_group_size=4 → 4个不同条件
                      学习条件依赖的策略
```

### 3. 计算效率

分离这两个参数便于优化：

```
显存不足？
- 减小 each_prompt_sample (如16→8)
- 保持 sample_group_size (如4)
- 结果: 64→32个分子，显存压力减半

想要更多条件多样性？
- 增大 sample_group_size (如4→6)
- 保持 each_prompt_sample (如16)
- 结果: 64→96个分子，条件更多样
```

---

## 数据流示意

### 配置示例：sample_group_size=4, each_prompt_sample=16

```
Step 1: 为4个group采样原子数
nodesxsample = [12, 18, 22, 28]  (4个不同的值)

Step 2: 创建group索引
group_index = [0, 1, 2, 3]

Step 3: 为每个group复制each_prompt_sample次
nodesxsample = [12,12,...,12(16次),
                18,18,...,18(16次),
                22,22,...,22(16次),
                28,28,...,28(16次)]

group_index = [0,0,...,0(16次),
               1,1,...,1(16次),
               2,2,...,2(16次),
               3,3,...,3(16次)]

最终batch_size = 4 × 16 = 64
```

**模型为64个分子生成结构：**
- 前16个: 12原子的分子 (group 0)
- 中间16个: 18原子的分子 (group 1)
- 后面16个: 22原子的分子 (group 2)
- 最后16个: 28原子的分子 (group 3)

**为64个分子计算奖励：**
- Group 0 (12原子): 16个分子的平均奖励
- Group 1 (18原子): 16个分子的平均奖励
- Group 2 (22原子): 16个分子的平均奖励
- Group 3 (28原子): 16个分子的平均奖励

**学习：** 不同原子数条件下的最优策略

---

## 实际应用

### 场景1：想要更多原子数多样性

```yaml
# 原配置
sample_group_size: 4      # 4个不同原子数
each_prompt_sample: 16    # 每个16个分子
# 总计: 64个分子

# 修改为
sample_group_size: 8      # 8个不同原子数
each_prompt_sample: 16    # 每个16个分子
# 总计: 128个分子
# 效果: 原子数条件更多样，模型学到更多条件依赖
```

### 场景2：想要更多同条件采样

```yaml
# 原配置
sample_group_size: 4      # 4个不同原子数
each_prompt_sample: 16    # 每个16个分子
# 总计: 64个分子

# 修改为
sample_group_size: 4      # 4个不同原子数
each_prompt_sample: 32    # 每个32个分子
# 总计: 128个分子
# 效果: 同条件下采样更多，奖励估计更准确
```

### 场景3：显存不足

```yaml
# 原配置
sample_group_size: 4      # 4个不同原子数
each_prompt_sample: 16    # 每个16个分子
# 总计: 64个分子

# 修改为
sample_group_size: 2      # 2个不同原子数
each_prompt_sample: 16    # 每个16个分子
# 总计: 32个分子
# 效果: 显存压力减半，但条件多样性降低
```

---

## 总结

| 参数 | 控制维度 | 影响 | 增大效果 |
|------|---------|------|---------|
| **sample_group_size** | 原子数条件多样性 | 学习条件依赖 | 模型学到更多条件下的最优策略 |
| **each_prompt_sample** | 同条件采样数 | 奖励估计准确性 | 同条件下样本更多，奖励估计更稳定 |

**关键理解：**
- `sample_group_size` = 有多少个"不同的原子数条件"
- `each_prompt_sample` = 每个条件生成多少个分子
- `batch_size` = `sample_group_size` × `each_prompt_sample` = 总分子数
