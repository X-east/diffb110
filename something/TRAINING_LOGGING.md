# DiffSBDD-DDPO 训练日志记录指南

## 概述

训练过程中会自动记录以下内容到输出目录。

## 输出目录结构

```
./exp/diffsbdd-ddpo-production/          # 正式训练输出
├── training_20260312_193520.log         # 训练日志文件
├── config.yaml                          # 保存的配置文件
├── checkpoints/                         # 模型checkpoint
│   ├── diffsbdd-ddpo-epoch=00-train_loss=0.0580.ckpt
│   ├── diffsbdd-ddpo-epoch=01-train_loss=0.0450.ckpt
│   └── last.ckpt                        # 最后一个checkpoint
└── lightning_logs/                      # TensorBoard日志
    └── version_0/
        ├── events.out.tfevents.*
        └── hparams.yaml

./log/ddp-4gpu-test/                     # 测试训练输出
├── training_*.log
├── config.yaml
├── checkpoints/
└── lightning_logs/
```

## 记录的指标

### 1. 训练损失 (Training Loss)
- **train_loss**: PPO策略梯度损失
- **train/Loss**: 每个timestep的平均损失
- **train/ClipFrac**: 策略梯度裁剪比例（PPO关键指标）

### 2. 分子生成指标
- **train/filtering_ratio**: 有效分子比例（1.0 = 100%有效）
- **train/novelty_penalty_ratio**: 新颖性惩罚比例

### 3. 奖励指标（每轮都会记录）
- **train/rewards**: 综合奖励分数（平均值）
- **train/docking_reward**: 对接得分（平均值，越低越好，范围通常 -10 ~ 0）
- **train/druglikeness_reward**: 药物相似性得分（平均值）
- **train/qed_reward**: QED得分（平均值，0-1，越高越好）
- **train/sa_reward**: SA得分（平均值，0-1，越高越好）

### 4. 分子统计
- **train/stability**: 分子稳定性（有效性）
- **train/num_atoms**: 生成分子的原子数统计

### 5. 优化器指标
- **train/learning_rate**: 当前学习率
- **train/grad_norm**: 梯度范数

## 采样和批处理详解

### 采样数量计算
```
batch_size = sample_group_size × each_prompt_sample
6GPU配置: batch_size = 3 × 24 = 72个分子/batch

配置文件中的 train.batch_size = 192 是指：
- 这是训练时的总batch大小（用于梯度累积或多step组合）
- 实际每个step生成的分子数 = 72
- 即：每个step生成72个分子，多个step的结果可以组合成192的训练batch
```

### 每轮记录的内容
**每个训练step都会记录：**
- 所有采样分子的docking得分（72个）
- 所有采样分子的QED得分（72个）
- 所有采样分子的SA得分（72个）
- 所有采样分子的综合奖励（72个）
- 这些得分的平均值、最小值、最大值

**每个epoch结束时会记录：**
- 该epoch所有step的平均指标
- 最佳分子的得分
- 最差分子的得分

## 日志文件内容

### training_*.log
包含详细的训练过程信息：

```
2026-03-12 19:35:20,255 - DiffSBDD-DDPO - INFO - Loading config from: ddpo/configs/ddp_4gpu_production.yaml
2026-03-12 19:35:20,260 - DiffSBDD-DDPO - INFO - Initializing DiffSBDDModel...
2026-03-12 19:35:20,470 - DiffSBDD-DDPO - INFO - Initializing DataLoader...
2026-03-12 19:35:20,470 - DiffSBDD-DDPO - INFO - Initializing Rewards...
...
Epoch 1 Summary:
  Average Loss: -0.0580
  ClipFrac: 0.0020
  filtering_ratio: 1.0000
  novelty_penalty_ratio: 1.0000
```

## Checkpoint 文件

### 保存策略
- **checkpoint_interval**: 每N个epoch保存一次
- **save_top_k**: 保存最好的K个checkpoint（按train_loss排序）
- **save_last**: 总是保存最后一个checkpoint

### 使用Checkpoint恢复训练

```bash
# 从最后一个checkpoint恢复
python ddpo/main_diffsbdd_lightning.py \
  --config_path ddpo/configs/ddp_4gpu_production.yaml \
  --resume \
  --checkpoint_path ./exp/diffsbdd-ddpo-production/checkpoints/last.ckpt
```

## 对接配置：CPU vs GPU

### UniDock GPU对接（推荐）
当前配置使用 **UniDock GPU版本** 进行对接：

```yaml
# 在 ddpo/configs/ddp_6gpu_production.yaml 中
reward:
  docking:
    exhaustiveness: 8  # GPU版本的详尽程度
```

**优势：**
- 速度快：10-100倍于CPU版本
- 支持批量对接
- 充分利用GPU资源
- 适合大规模分子生成

**工作流程：**
1. 生成分子（GPU）→ 72个分子/step
2. 转换为PDBQT格式（CPU）
3. UniDock GPU对接（GPU）→ 并行计算所有分子
4. 返回对接得分

### CPU对接（Smina，已弃用）
旧配置使用Ray + CPU对接，现已替换为GPU版本。

---

## 详细得分记录机制

### 每个Step记录的所有得分

当训练运行时，**每个step都会记录所有分子的详细得分**：

```
Step 1:
  采样72个分子
  ├─ 分子1: docking=-8.5, qed=0.72, sa=0.65, reward=0.68
  ├─ 分子2: docking=-7.2, qed=0.68, sa=0.70, reward=0.62
  ├─ ...
  └─ 分子72: docking=-6.1, qed=0.75, sa=0.68, reward=0.65
  
  平均值: docking=-7.3, qed=0.71, sa=0.68, reward=0.64
  最好: docking=-9.2, qed=0.78, sa=0.72, reward=0.72
  最差: docking=-5.1, qed=0.62, sa=0.58, reward=0.55
```

### 记录的得分类型

| 得分类型 | 范围 | 含义 | 记录位置 |
|---------|------|------|---------|
| **docking_reward** | -10~0 | 对接亲和力，越低越好 | data.batch['docking_reward'] |
| **qed_reward** | 0-1 | 药物相似性，越高越好 | data.batch['qed_reward'] |
| **sa_reward** | 0-1 | 合成可行性，越高越好 | data.batch['sa_reward'] |
| **druglikeness_reward** | 0-1 | QED+SA的加权平均 | data.batch['druglikeness_reward'] |
| **rewards** | 0-1 | 最终综合奖励 | data.batch['rewards'] |

### 如何访问这些得分

在 `CompositeReward.calculate_rewards()` 中，所有得分都被保存到 `data.batch` 中：

```python
# 对接得分
data.batch['docking_reward']      # shape: [72]

# 药物相似性得分
data.batch['druglikeness_reward'] # shape: [72]

# 综合奖励
data.batch['rewards']             # shape: [72]
```

### 每轮（Epoch）的平均得分

在 `on_train_epoch_end()` 中，Lightning会自动计算并记录：

```
Epoch 1 Summary:
  Average Loss: -0.0580
  ClipFrac: 0.0020
  filtering_ratio: 1.0000
  novelty_penalty_ratio: 1.0000
  docking_reward: -7.45 (平均对接得分)
  qed_reward: 0.71 (平均QED)
  sa_reward: 0.68 (平均SA)
  druglikeness_reward: 0.695 (平均药物相似性)
  rewards: 0.64 (平均综合奖励)
```

### TensorBoard中的可视化

所有这些指标都会被记录到TensorBoard中，可以查看：

```bash
tensorboard --logdir ./exp/diffsbdd-ddpo-6gpu-production/lightning_logs
```

在TensorBoard中可以看到：
- 每个step的所有得分曲线
- 每个epoch的平均得分
- 得分的分布变化趋势

## 日志查看命令

```bash
# 查看最新日志（实时跟踪）
tail -f ./exp/diffsbdd-ddpo-6gpu-production/training_*.log

# 查看最后100行
tail -100 ./exp/diffsbdd-ddpo-6gpu-production/training_*.log

# 搜索特定内容
grep "Epoch" ./exp/diffsbdd-ddpo-6gpu-production/training_*.log

# 查看所有checkpoint
ls -lh ./exp/diffsbdd-ddpo-6gpu-production/checkpoints/

# 查看配置
cat ./exp/diffsbdd-ddpo-6gpu-production/config.yaml
```

## 导出所有分子得分

### 当前记录方式

所有分子的得分已经被记录在内存中，通过以下方式访问：

**在训练过程中：**
```python
# 在 lightning_ddpo_module.py 的 training_step() 中
# 所有得分都在 data.batch 中：
data.batch['docking_reward']      # 所有72个分子的对接得分
data.batch['qed_reward']          # 所有72个分子的QED得分
data.batch['sa_reward']           # 所有72个分子的SA得分
data.batch['druglikeness_reward'] # 所有72个分子的药物相似性得分
data.batch['rewards']             # 所有72个分子的综合奖励
```

### 导出到CSV的方法

**方法1：修改训练代码保存得分**

在 `ddpo/trainer/lightning_ddpo_module.py` 的 `training_step()` 中添加：

```python
import pandas as pd
from pathlib import Path

def training_step(self, batch, batch_idx):
    loss, metrics = self.forward(batch)
    
    # 保存所有分子的得分到CSV
    if 'docking_reward' in batch and self.trainer.is_global_zero:
        scores_dir = Path(self.config['output']['dir']) / 'molecule_scores'
        scores_dir.mkdir(exist_ok=True)
        
        scores_df = pd.DataFrame({
            'docking': batch['docking_reward'].cpu().numpy(),
            'qed': batch['qed_reward'].cpu().numpy(),
            'sa': batch['sa_reward'].cpu().numpy(),
            'druglikeness': batch['druglikeness_reward'].cpu().numpy(),
            'reward': batch['rewards'].cpu().numpy()
        })
        
        csv_file = scores_dir / f'epoch_{self.current_epoch:03d}_step_{batch_idx:04d}.csv'
        scores_df.to_csv(csv_file, index=False)
    
    # ... 其他代码
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    return None
```

**方法2：从TensorBoard导出**

```bash
# 导出所有标量指标为CSV
tensorboard --logdir ./exp/diffsbdd-ddpo-6gpu-production/lightning_logs \
  --export_dir ./metrics_export
```

### 得分统计分析

**Python脚本分析所有得分：**

```python
import pandas as pd
import numpy as np
from pathlib import Path

# 读取所有CSV文件
scores_dir = Path('./exp/diffsbdd-ddpo-6gpu-production/molecule_scores')
all_scores = []

for csv_file in sorted(scores_dir.glob('*.csv')):
    df = pd.read_csv(csv_file)
    df['epoch'] = int(csv_file.stem.split('_')[1])
    df['step'] = int(csv_file.stem.split('_')[3])
    all_scores.append(df)

combined_df = pd.concat(all_scores, ignore_index=True)

# 统计分析
print("=== 全局统计 ===")
print(combined_df[['docking', 'qed', 'sa', 'reward']].describe())

# 每个epoch的平均得分
print("\n=== 每个Epoch的平均得分 ===")
epoch_stats = combined_df.groupby('epoch')[['docking', 'qed', 'sa', 'reward']].mean()
print(epoch_stats)

# 找出最好的分子
print("\n=== 最好的分子 ===")
best_idx = combined_df['reward'].idxmax()
print(combined_df.iloc[best_idx])

# 找出最差的分子
print("\n=== 最差的分子 ===")
worst_idx = combined_df['reward'].idxmin()
print(combined_df.iloc[worst_idx])

# 保存统计结果
combined_df.to_csv('./all_molecule_scores.csv', index=False)
epoch_stats.to_csv('./epoch_average_scores.csv')
```

### 得分分布可视化

```python
import matplotlib.pyplot as plt

# 绘制得分分布
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(combined_df['docking'], bins=30, edgecolor='black')
axes[0, 0].set_title('Docking Score Distribution')
axes[0, 0].set_xlabel('Docking Score')

axes[0, 1].hist(combined_df['qed'], bins=30, edgecolor='black')
axes[0, 1].set_title('QED Score Distribution')
axes[0, 1].set_xlabel('QED Score')

axes[1, 0].hist(combined_df['sa'], bins=30, edgecolor='black')
axes[1, 0].set_title('SA Score Distribution')
axes[1, 0].set_xlabel('SA Score')

axes[1, 1].hist(combined_df['reward'], bins=30, edgecolor='black')
axes[1, 1].set_title('Combined Reward Distribution')
axes[1, 1].set_xlabel('Reward')

plt.tight_layout()
plt.savefig('./score_distributions.png', dpi=150)
plt.show()

# 绘制得分随epoch的变化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, col in zip(axes.flat, ['docking', 'qed', 'sa', 'reward']):
    epoch_stats[col].plot(ax=ax, marker='o')
    ax.set_title(f'{col.capitalize()} Score Trend')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{col.capitalize()} Score')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./score_trends.png', dpi=150)
plt.show()
```

## 关键指标解释

| 指标 | 范围 | 含义 |
|------|------|------|
| train_loss | 负数 | PPO损失，越低越好 |
| ClipFrac | 0-1 | 策略更新被裁剪的比例，通常 < 0.1 |
| filtering_ratio | 0-1 | 有效分子比例，1.0最好 |
| docking_reward | -10~0 | 对接得分，越低（更负）越好 |
| druglikeness_reward | 0-1 | 药物相似性，越高越好 |
| learning_rate | 正数 | 当前学习率 |

## 常见问题

### Q: 为什么没有保存checkpoint？
A: 检查 `checkpoint_interval` 是否大于总epoch数。

### Q: 如何恢复中断的训练？
A: 使用 `--resume` 和 `--checkpoint_path` 参数从最后一个checkpoint恢复。

### Q: 日志文件在哪里？
A: 在输出目录的 `training_*.log` 文件中，文件名包含时间戳。

### Q: 如何修改日志级别？
A: 在运行脚本时使用 `--log_level DEBUG|INFO|WARNING|ERROR` 参数。

### Q: 总采样数是72，但batch_size是192，这是怎么回事？
A: 
- **实际采样数 = sample_group_size × each_prompt_sample = 3 × 24 = 72**
- 配置文件中的 `train.batch_size = 192` 是指训练时的总batch大小
- 这是用于梯度累积或多step组合的参数
- 每个step实际生成72个分子，多个step的结果可以组合成192的训练batch

### Q: 所有分子的得分都被记录了吗？
A: 
- **是的**，所有72个分子的得分都被记录在 `data.batch` 中
- 包括：docking_reward, qed_reward, sa_reward, druglikeness_reward, rewards
- 每个step都会记录这些得分
- 可以通过修改训练代码导出到CSV进行分析

### Q: 如何查看每轮的平均得分？
A: 
- 在日志文件中搜索 "Epoch X Summary"
- 或在TensorBoard中查看 train/docking_reward, train/qed_reward 等指标
- 或使用提供的Python脚本分析导出的CSV文件

### Q: UniDock GPU对接和CPU对接有什么区别？
A:
- **UniDock GPU**（当前使用）：快10-100倍，支持批量对接，充分利用GPU
- **Smina CPU**（已弃用）：速度慢，但更精确，使用Ray并行化
- 当前配置已完全切换到UniDock GPU版本

### Q: 如何修改对接的详尽程度？
A: 在配置文件中修改 `reward.docking.exhaustiveness`
- 值越大越精确但越慢（范围1-8）
- 当前设置为8（最详尽）

### Q: 能否只记录最好的分子而不是所有分子？
A: 可以，修改训练代码在 `on_train_epoch_end()` 中只保存top-k分子的得分
