# RLPF: 基于物理反馈的强化学习

一个灵活的框架，用于使用扩散模型实现基于物理反馈的强化学习，构建于 verl_diffusion 框架之上。

## 概述

RLPF（Reinforcement Learning with Physical Feedback，基于物理反馈的强化学习）框架提供了一种结构化的方法，用于实现具有强化学习能力和物理反馈机制的扩散模型。它被设计为模块化和可扩展的，允许您集成自己的模型、奖励函数和训练过程，同时利用稳健的 verl_diffusion 基础。

## 框架结构

RLPF 框架基于 verl_diffusion 架构，组织为几个关键组件：

- **Model（模型）**: 定义扩散模型架构
- **Trainer（训练器）**: 处理训练过程
- **Worker（工作器）**: 包含 rollout、奖励和过滤的组件
- **Utils（工具）**: 辅助函数和实用程序
- **Dataloader（数据加载器）**: 数据加载和预处理

## 📦 安装

### 1. 设置 Python 环境

```bash
conda create -n RLPF python=3.10.14
conda activate RLPF
pip install -r requirements.txt
```

### 2. 安装 xtb 用于力计算（用于奖励）

```bash
cd RLPF
git clone https://github.com/grimme-lab/xtb-python.git
cd xtb-python
conda install mkl mkl-devel
conda install -c conda-forge "gfortran<12"
conda install -c conda-forge mkl mkl-devel blas lapack
pip install .
cd ..
```

## 🚀 如何使用 RLPF 用 xtb 奖励微调 EDM

### 1. 设置 EDM 环境
```bash
cd ./Model/EDM
pip install .
```

### 2. QM9 分子生成
```bash
bash example/edm_ddpo_xtb/run.sh
```

### 3. QM9 分子生成评估
```bash
bash example/edm_ddpo_xtb/run_eval.sh
```

## RLPF 入门指南

RLPF 框架允许您使用物理反馈奖励（如 XTB（扩展紧束缚）计算）来微调扩散模型（如 EDM）。以下是入门方法：

### 1. 定义您的自定义模型

要在 RLPF 框架中使用您自己的模型，您需要创建一个继承自 verl_diffusion 包中 `BaseModel` 的类：

```python
from verl_diffusion.model.base import BaseModel

class YourCustomModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        # 初始化您的模型组件
        
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context=None, fix_noise=False, timestep=1000):
        """
        从您的生成模型中抽取样本。
        
        参数:
            n_samples: 要生成的样本数量
            n_nodes: 每个样本的节点数
            node_mask: 节点掩码
            edge_mask: 边掩码
            context: 可选的上下文信息
            fix_noise: 是否使用固定噪声
            timestep: 扩散时间步数
            
        返回:
            生成的样本和相关信息
        """
        # 实现您的采样逻辑
        pass
        
    def compute_log_p_zs_given_zt(self, x, mu, sigma, node_mask=None):
        """
        计算给定 zt 时 zs 的对数概率。
        
        参数:
            x: 输入张量
            mu: 均值张量
            sigma: 标准差张量
            node_mask: 可选的节点掩码
            
        返回:
            对数概率值
        """
        # 实现您的概率计算
        pass
        
    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context=None, fix_noise=False):
        """
        从 p(zs | zt) 中采样。
        
        参数:
            s: 源时间步
            t: 目标时间步
            zt: 时间步 t 的潜变量
            node_mask: 节点掩码
            edge_mask: 边掩码
            context: 可选的上下文
            fix_noise: 是否使用固定噪声
            
        返回:
            采样值和相关信息
        """
        # 实现您的条件采样
        pass
```

### 2. 定义您的物理反馈奖励函数

创建一个继承自基础奖励类的奖励函数：

```python
from verl_diffusion.worker.reward.base import BaseReward

class YourCustomReward(BaseReward):
    def __init__(self, config):
        super().__init__(config)
        # 初始化您的奖励组件
        
    def calculate_rewards(self, samples):
        """
        计算生成样本的奖励。
        
        参数:
            samples: 生成的样本
            
        返回:
            奖励值
        """
        # 实现您的奖励计算
        pass
```

### 3. 配置您的 RLPF 训练过程

创建一个配置文件，指定您的 RLPF 训练参数和物理反馈设置：

```python
config = {
    "model": {
        "diffusion_steps": 1000,           # 扩散步数
        "diffusion_noise_schedule": "cosine",  # 噪声调度
        "diffusion_noise_precision": 1e-5,     # 噪声精度
        "diffusion_loss_type": "l2",           # 损失类型
        "normalize_factors": [1.0, 1.0, 1.0],  # 归一化因子
        "include_charges": True                 # 是否包含电荷
    },
    "train": {
        "batch_size": 32,              # 批次大小
        "micro_batch_size": 8,         # 微批次大小
        "learning_rate": 1e-4,         # 学习率
        "clip_advantage_value": 5.0,   # 优势裁剪值
        "save_path": "./exp/your_model"  # 保存路径
    },
    "dataloader": {
        "epoches": 100                 # 训练轮数
    },
    "wandb": {
        "enabled": True,               # 是否启用 wandb
        "project": "your-project",     # 项目名
        "name": "your-run-name"        # 运行名称
    }
}
```

### 4. 设置 RLPF 训练流水线

```python
from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer
from verl_diffusion.dataloader.dataloader import EDMDataLoader

# 初始化您的组件
model = YourCustomModel(config)
dataloader = EDMDataLoader(config)
rollout = YourCustomRollout(config, model)
rewarder = YourCustomReward(config)
actor = YourCustomActor(config)

# 创建训练器
trainer = DDPOTrainer(
    config=config,
    model=model,
    dataset_info=dataset_info,
    device=device,
    dataloader=dataloader,
    rollout=rollout,
    rewarder=rewarder,
    actor=actor
)

# 开始训练
trainer.fit()
```

## RLPF 的关键组件

RLPF 框架利用 verl_diffusion 的以下关键组件，同时添加物理反馈能力：

### DataProto（数据协议）

框架使用 `DataProto` 在组件之间进行数据交换。它提供了一种标准化的方式来处理张量和非张量数据：

```python
from verl_diffusion.protocol import DataProto

# 创建 DataProto
data = DataProto.from_dict(
    tensors={"key1": tensor1, "key2": tensor2},
    non_tensors={"key3": array1},
    meta_info={"info": "metadata"}
)

# 访问数据
tensor_data = data.batch["key1"]           # 张量数据
non_tensor_data = data.non_tensor_batch["key3"]  # 非张量数据
meta_data = data.meta_info["info"]         # 元数据
```

### Rollout（轨迹采样）

rollout 组件处理生成过程：

```python
from verl_diffusion.worker.rollout.base import BaseRollout

class YourCustomRollout(BaseRollout):
    def generate_minibatch(self, batch):
        """
        为小批次生成样本。
        
        参数:
            batch: 输入批次数据
            
        返回:
            生成的样本
        """
        # 实现您的生成逻辑
        pass
```

### Actor（策略网络）

actor 组件定义动作选择的策略：

```python
from verl_diffusion.worker.actor.base import BaseActor

class YourCustomActor(BaseActor):
    def __init__(self, config):
        super().__init__(config)
        # 初始化您的 actor 组件
        
    def act(self, state):
        """
        基于当前状态选择动作。
        
        参数:
            state: 当前状态
            
        返回:
            选择的动作
        """
        # 实现您的动作选择逻辑
        pass
```

## 高级用法

### 自定义过滤器

您可以实现自定义过滤器来处理生成的样本：

```python
from verl_diffusion.worker.filter.base import BaseFilter

class YourCustomFilter(BaseFilter):
    def __init__(self, config):
        super().__init__(config)
        # 初始化您的过滤器组件
        
    def filter(self, samples):
        """
        过滤生成的样本。
        
        参数:
            samples: 生成的样本
            
        返回:
            过滤后的样本
        """
        # 实现您的过滤逻辑
        pass
```

### 并行处理

框架支持使用 Ray 进行并行处理：

```python
import ray

# 初始化 Ray
ray.init()

# 在您的组件中使用 Ray 进行并行处理
@ray.remote
def parallel_process(data):
    # 并行处理数据
    pass
```

## RLPF 最佳实践

1. **模型设计**:
   - 确保您的模型实现了 `BaseModel` 的所有必需方法
   - 对可变大小的输入使用适当的掩码
   - 实现正确的归一化和反归一化

2. **物理反馈奖励函数**:
   - 设计包含有意义物理属性的奖励（例如 XTB 能量、稳定性）
   - 归一化奖励以防止训练不稳定
   - 考虑使用奖励塑形以更好地学习物理约束

3. **RLPF 训练过程**:
   - 为您的硬件使用适当的批次大小
   - 使用 wandb 监控训练指标
   - 实现正确的检查点和模型保存
   - 平衡物理反馈频率与计算成本

4. **数据处理**:
   - 使用 `DataProto` 进行一致的数据交换
   - 实现正确的批处理和分块
   - 使用适当的掩码处理可变大小的输入

## 故障排除

常见问题和解决方案：

1. **内存不足**:
   - 减小批次大小
   - 使用梯度累积
   - 实现正确的内存管理

2. **训练不稳定**:
   - 检查奖励归一化
   - 调整学习率
   - 监控优势裁剪

3. **训练缓慢**:
   - 使用适当的批次大小
   - 实现并行处理
   - 优化数据加载

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 LICENSE 文件。

---

## 术语表（面向非计算机专业）

| 英文术语 | 中文 | 解释 |
|---------|------|------|
| Reinforcement Learning | 强化学习 | 通过奖励信号学习最优策略的机器学习方法 |
| Physical Feedback | 物理反馈 | 基于物理计算（如能量、力）的奖励信号 |
| Diffusion Model | 扩散模型 | 通过逐步去噪生成数据的生成模型 |
| DDPO | 去噪扩散策略优化 | 用于微调扩散模型的强化学习算法 |
| XTB | 扩展紧束缚 | 一种快速量子化学计算方法 |
| Rollout | 轨迹采样 | 生成完整的采样轨迹 |
| Actor | 策略网络 | 决定采取什么动作的神经网络 |
| Reward | 奖励 | 评估生成样本质量的分数 |
| Filter | 过滤器 | 筛选有效/无效样本的组件 |
| DataProto | 数据协议 | 组件间数据交换的标准格式 |
| Batch Size | 批次大小 | 一次处理的样本数量 |
| Learning Rate | 学习率 | 模型参数更新的步长 |
| Checkpoint | 检查点 | 保存的模型状态 |
| wandb | Weights & Biases | 实验追踪和可视化平台 |
| Ray | Ray | 分布式计算框架 |
| Gradient Accumulation | 梯度累积 | 多个小批次累积梯度后再更新 |
| Advantage Clipping | 优势裁剪 | PPO 算法中限制策略更新幅度的技术 |

---

## RLPF 与 DiffSBDD 集成说明

在本项目中，我们将 RLPF 框架与 DiffSBDD 模型集成，用于优化针对 RE-CmeB 蛋白的分子生成：

| RLPF 组件 | 本项目实现 | 说明 |
|-----------|-----------|------|
| BaseModel | DiffSBDDModel | 封装 DiffSBDD 的 LigandPocketDDPM |
| BaseReward | DockingReward | 使用 AutoDock Vina 计算对接分数 |
| BaseReward | DruglikenessReward | 计算 QED 和 SA 分数 |
| BaseReward | CompositeReward | 组合多个奖励函数 |
| BaseFilter | DiffSBDDFilter | 使用 DiffSBDD 的 build_molecule 验证有效性 |
| DDPOTrainer | 直接复用 | DDPO 训练循环 |
| EDMActor | 直接复用 | PPO 策略梯度更新 |
