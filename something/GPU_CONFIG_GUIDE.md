# GPU配置调整指南

## 快速对比

| 参数 | 4GPU | 6GPU | 8GPU |
|------|------|------|------|
| num_gpus | 4 | 6 | 8 |
| batch_size | 64 | 192 | 256 |
| per_device_batch_size | 16 | 32 | 32 |
| sample_group_size | 2 | 3 | 4 |
| each_prompt_sample | 8 | 24 | 32 |
| num_cpus (Ray) | 8 | 18 | 24 |
| output_dir | ./log/ddp-4gpu-test | ./exp/diffsbdd-ddpo-6gpu-production | ./exp/diffsbdd-ddpo-production |

## 配置文件位置

```bash
# 4GPU 测试配置
ddpo/configs/ddp_4gpu_test.yaml

# 4GPU 正式配置
ddpo/configs/ddp_4gpu_production.yaml

# 6GPU 正式配置（新建）
ddpo/configs/ddp_6gpu_production.yaml

# 8GPU 正式配置
ddpo/configs/ddp_gpu_production.yaml
```

## 关键参数说明

### 1. num_gpus
- 使用的GPU数量
- 直接影响DDP并行度
- **改为6GPU**: `num_gpus: 6`

### 2. batch_size
- 总batch size = num_gpus × per_device_batch_size
- 4GPU: 4 × 16 = 64
- 6GPU: 6 × 32 = 192
- 8GPU: 8 × 32 = 256
- **改为6GPU**: `batch_size: 192`

### 3. per_device_batch_size
- 每个GPU上的batch size
- 受GPU显存限制
- RTX 3080 (10GB) 通常可以支持 32
- **改为6GPU**: `per_device_batch_size: 32`

### 4. sample_group_size
- 采样组大小
- 通常 = num_gpus / 2（向下取整）
- 4GPU: 2, 6GPU: 3, 8GPU: 4
- **改为6GPU**: `sample_group_size: 3`

### 5. each_prompt_sample
- 每个prompt采样数
- 通常 = sample_group_size × 8
- 4GPU: 2×8=16, 6GPU: 3×8=24, 8GPU: 4×8=32
- **改为6GPU**: `each_prompt_sample: 24`

### 6. num_cpus (Ray)
- Ray并行对接使用的CPU核心数
- 通常 = num_gpus × 3 ~ 4
- 4GPU: 8, 6GPU: 18, 8GPU: 24
- **改为6GPU**: `num_cpus: 18`

## 运行6GPU训练

### 方法1: 修改现有脚本
编辑 `run_production_training.sh`，改为：

```bash
CONFIG_PATH="ddpo/configs/ddp_6gpu_production.yaml"
NUM_GPUS=6
```

### 方法2: 直接运行命令
```bash
cd /home/md666/data/disk6/DiffB110

MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
torchrun --nproc_per_node=6 \
  ddpo/main_diffsbdd_lightning.py \
  --config_path ddpo/configs/ddp_6gpu_production.yaml \
  --seed 42 \
  --log_level INFO
```

## 性能对比

### 理论吞吐量
- 4GPU: 64 samples/batch
- 6GPU: 192 samples/batch (3倍)
- 8GPU: 256 samples/batch (4倍)

### 显存使用
- 4GPU: ~8GB per GPU
- 6GPU: ~9GB per GPU
- 8GPU: ~9GB per GPU

### 训练时间
- 6GPU相比4GPU: 快约 1.5-2 倍
- 6GPU相比8GPU: 慢约 20-30%

## 自定义GPU数量

如果要使用其他GPU数量（如2GPU、3GPU等），按以下公式调整：

```
batch_size = num_gpus × per_device_batch_size
sample_group_size = num_gpus / 2 (向下取整)
each_prompt_sample = sample_group_size × 8
num_cpus = num_gpus × 3
```

### 例子：2GPU配置
```yaml
ddp:
  num_gpus: 2

train:
  batch_size: 32          # 2 × 16
  per_device_batch_size: 16

dataloader:
  sample_group_size: 1    # 2 / 2
  each_prompt_sample: 8   # 1 × 8

ray:
  num_cpus: 6             # 2 × 3
```

## 常见问题

### Q: 为什么batch_size要这样设置？
A: batch_size应该是num_gpus的倍数，这样每个GPU分配的数据量相等。

### Q: 能否只改num_gpus不改其他参数？
A: 可以，但性能不会最优。建议按比例调整所有参数。

### Q: 6GPU比4GPU快多少？
A: 理论上快1.5倍，实际取决于通信开销和显存限制。

### Q: 显存不足怎么办？
A: 减小 `per_device_batch_size` 或 `each_prompt_sample`。

### Q: 如何检查GPU是否都被使用？
A: 运行时执行 `nvidia-smi` 查看所有GPU的利用率。
