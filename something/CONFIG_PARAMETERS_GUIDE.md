# DiffSBDD-DDPO 配置参数详解

## 1. 顶层参数


| 参数     | 示例值                  | 含义     | 作用              |
| ------ | -------------------- | ------ | --------------- |
| `name` | `diffsbdd-ddpo-8gpu` | 训练任务名称 | 用于标识实验，区分不同训练任务 |


---

## 2. 模型配置 (model)


| 参数           | 示例值                                                     | 含义      | 作用                     |
| ------------ | ------------------------------------------------------- | ------- | ---------------------- |
| `checkpoint` | `./DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt` | 预训练模型路径 | 加载初始模型权重进行微调           |
| `time_step`  | 500                                                     | 扩散去噪步数  | 越大生成越精细，计算量越大          |
| `device`     | `cuda`                                                  | 运行设备    | `cuda`使用GPU，`cpu`使用CPU |


---

## 3. 口袋配置 (pocket)


| 参数           | 示例值                     | 含义         | 作用                  |
| ------------ | ----------------------- | ---------- | ------------------- |
| `pdb_path`   | `./data/8GJJ-C-Cip.pdb` | 蛋白质PDB文件路径 | 提供蛋白质结构信息，指导条件生成    |
| `ref_ligand` | `C:1101`                | 参考配体标识     | 格式为"链:残基号"，用于提取配体信息 |


---

## 4. 奖励函数配置 (reward)

### 4.1 权重 (weights)


| 参数             | 示例值 | 含义     | 作用           |
| -------------- | --- | ------ | ------------ |
| `docking`      | 0.7 | 对接得分权重 | 总奖励中对接得分占70% |
| `druglikeness` | 0.3 | 类药性权重  | 总奖励中类药性占30%  |


### 4.2 对接参数 (docking)


| 参数                     | 示例值    | 含义          | 作用                    |
| ---------------------- | ------ | ----------- | --------------------- |
| `normalization_factor` | -10.0  | 对接得分归一化因子   | 将原始得分(如-12~0)归一化到合理范围 |
| `center_x`             | 161.45 | 对接盒中心X坐标(Å) | 定义搜索空间中心              |
| `center_y`             | 169.80 | 对接盒中心Y坐标(Å) | 定义搜索空间中心              |
| `center_z`             | 174.23 | 对接盒中心Z坐标(Å) | 定义搜索空间中心              |
| `size_x`               | 20.0   | 对接盒X维尺寸(Å)  | 定义搜索空间大小              |
| `size_y`               | 20.0   | 对接盒Y维尺寸(Å)  | 定义搜索空间大小              |
| `size_z`               | 20.0   | 对接盒Z维尺寸(Å)  | 定义搜索空间大小              |
| `exhaustiveness`       | 8      | 对接详尽程度      | 1-8，越大越精确但越慢          |
| `chunk_size`           | 16     | 批量对接分子数     | 一次提交给对接队列的分子数         |


---

## 5. 数据加载器配置 (dataloader)


| 参数                   | 示例值 | 含义    | 作用           |
| -------------------- | --- | ----- | ------------ |
| `sample_group_size`  | 4   | 采样组数  | 同时处理的条件/口袋组数 |
| `each_prompt_sample` | 16  | 每组采样数 | 每个条件生成的分子数   |
| `epochs`             | 100 | 训练轮数  | 总共训练多少个epoch |
| `min_nodes`          | 10  | 最小原子数 | 生成分子的最少原子数   |
| `max_nodes`          | 30  | 最大原子数 | 生成分子的最多原子数   |


**采样计算：** `batch_size = sample_group_size × each_prompt_sample`

---

## 6. 训练参数 (train)

### 6.1 Batch相关


| 参数                       | 示例值 | 含义           | 作用             |
| ------------------------ | --- | ------------ | -------------- |
| `batch_size`             | 128 | 全局batch大小    | 所有GPU并行处理的总样本数 |
| `per_device_batch_size`  | 16  | 每GPU batch大小 | 每块GPU处理的样本数    |
| `train_micro_batch_size` | 8   | 微batch大小     | 用于梯度累积，实现显存优化  |


**关系：** `batch_size = num_gpus × per_device_batch_size`

### 6.2 优化器参数 (Adam)


| 参数                  | 示例值    | 含义      | 作用                 |
| ------------------- | ------ | ------- | ------------------ |
| `learning_rate`     | 3.0e-4 | 初始学习率   | 控制参数更新步长           |
| `adam_beta1`        | 0.9    | 一阶矩衰减率  | 动量项系数，通常0.9        |
| `adam_beta2`        | 0.999  | 二阶矩衰减率  | RMSprop项系数，通常0.999 |
| `adam_weight_decay` | 1.0e-4 | L2正则化系数 | 防止过拟合              |
| `adam_epsilon`      | 1.0e-8 | 数值稳定性常数 | 防止除零错误             |


### 6.3 PPO相关


| 参数              | 示例值  | 含义        | 作用                        |
| --------------- | ---- | --------- | ------------------------- |
| `clip_range`    | 0.2  | PPO裁剪范围   | 限制新旧策略比值在[0.8,1.2]，防止更新过大 |
| `adv_clip_max`  | 10.0 | 优势函数裁剪最大值 | 将优势函数限制在[-10,10]，稳定训练     |
| `max_grad_norm` | 1.0  | 梯度裁剪最大范数  | 防止梯度爆炸                    |


### 6.4 Checkpoint和日志


| 参数                    | 示例值 | 含义             | 作用             |
| --------------------- | --- | -------------- | -------------- |
| `checkpoint_interval` | 10  | checkpoint保存间隔 | 每N个epoch保存一次模型 |
| `log_interval`        | 1   | 日志记录间隔         | 每N个epoch记录一次指标 |


---

## 7. 分布式数据并行配置 (ddp)


| 参数                       | 示例值   | 含义          | 作用                              |
| ------------------------ | ----- | ----------- | ------------------------------- |
| `enabled`                | true  | 是否启用DDP     | true启用多GPU训练                    |
| `num_gpus`               | 8     | GPU数量       | 使用的GPU设备数                       |
| `strategy`               | `ddp` | 分布式策略       | PyTorch的DistributedDataParallel |
| `find_unused_parameters` | true  | 查找未使用参数     | 处理条件分支中的未使用参数                   |
| `sync_batchnorm`         | false | 同步BatchNorm | true时所有GPU共享BN统计量               |


---

## 8. Ray分布式计算配置 (ray)


| 参数         | 示例值 | 含义     | 作用                  |
| ---------- | --- | ------ | ------------------- |
| `num_cpus` | 16  | CPU核心数 | 分配给Ray的CPU核心，用于并行对接 |
| `num_gpus` | 0   | GPU数量  | Ray不使用GPU（对接用CPU）   |


**说明：** Ray用于并行执行分子对接任务，CPU核心数应与`chunk_size`匹配

---

## 9. Weights & Biases配置 (wandb)


| 参数        | 示例值        | 含义        | 作用                          |
| --------- | ---------- | --------- | --------------------------- |
| `enabled` | false      | 是否启用WandB | false不启用在线实验跟踪              |
| `mode`    | `disabled` | WandB运行模式 | `disabled`完全禁用，`online`在线记录 |


---

## 10. 输出配置 (output)


| 参数               | 示例值                       | 含义    | 作用                   |
| ---------------- | ------------------------- | ----- | -------------------- |
| `dir`            | `./log/ddp-8gpu-training` | 输出根目录 | 保存日志、checkpoint、分子结构 |
| `save_frequency` | 5                         | 保存频率  | 每N个epoch保存一次中间结果     |
| `metrics_file`   | `metrics.csv`             | 指标文件名 | 保存训练指标的CSV文件         |
| `molecules_dir`  | `molecules`               | 分子目录名 | 保存生成分子结构的子目录         |
| `verbose`        | true                      | 详细输出  | true输出详细日志信息         |


---

## 快速参考

### 常见修改场景


| 场景      | 修改参数                     | 建议值          |
| ------- | ------------------------ | ------------ |
| 显存不足    | `per_device_batch_size`  | 减小（如8或12）    |
| 想要更快训练  | `time_step`              | 减小（如100-200） |
| 想要更好质量  | `time_step`              | 增大（如1000）    |
| 想要更多采样  | `each_prompt_sample`     | 增大（如32）      |
| 想要更精确对接 | `exhaustiveness`         | 增大（如8）       |
| 想要更快对接  | `exhaustiveness`         | 减小（如4）       |
| 换不同蛋白质  | `pdb_path`, `ref_ligand` | 更新为新蛋白质      |


---

## 配置验证

运行前检查清单：

- `checkpoint` 文件存在
- `pdb_path` 文件存在
- `batch_size` = `num_gpus` × `per_device_batch_size`
- `sample_group_size` × `each_prompt_sample` 能整除 `batch_size`
- `num_cpus` ≥ `chunk_size`
- `output.dir` 目录可写
- GPU数量 ≥ `num_gpus`

