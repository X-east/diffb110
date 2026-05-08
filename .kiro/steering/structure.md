# 项目架构与结构

## 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                   ddpo/ (集成主代码)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐   ┌────────────────────────────────┐  │
│  │ DiffSBDDModel        │   │ EDMActor (RLPF)                │  │
│  │ (BaseModel)          │   │                                │  │
│  │ - sample()           │   │ - calculate_loss()             │  │
│  │ - sample_p_zs_given  │   │   PPO clip + kl_coef × KL     │  │
│  │   _zt()              │   │ - ref_model (deepcopy, 冻结)   │  │
│  │ - compute_log_prob() │   │                                │  │
│  │ - cached_pocket      │   └────────────────────────────────┘  │
│  └──────────┬───────────┘                                       │
│             │                                                     │
│  ┌──────────┴───────────────────────────────────────────────┐   │
│  │                    奖励系统（4层解耦）                       │   │
│  │                                                           │   │
│  │  docking.py ─→ docking_preparation.py ─→ docking_config.py│   │
│  │       │              (加氢+Gasteiger)       (常量)        │   │
│  │       │                                                   │   │
│  │  docking_registry.py ─→ docking_backends/                 │   │
│  │       (注册选择)         ├── unidock.py    (主后端)       │   │
│  │                          ├── autodock_gpu.py (备选)       │   │
│  │                          └── smina.py       (CPU备用)     │   │
│  │                                                           │   │
│  │  composite.py ─→ qed_reward.py + sa_reward.py            │   │
│  │  normalization.py ─→ z-score RewardNormalizer            │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ molecule_semantics.py (统一分子管线)                        │  │
│  │  - build_canonical_rdmol_from_sample()  构建+过滤         │  │
│  │  - prepare_mol_for_docking()            加氢+Gasteiger    │  │
│  │  - write_rdmol_to_sdf()                 SDF 写入           │  │
│  │  - write_rdmol_to_pdbqt()               SDF→obabel→PDBQT  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────┐   ┌───────────────────────────────┐   │
│  │ runtime/             │   │ kl/                           │   │
│  │ - build.py (工厂)    │   │ - gaussian_kl.py (纯张量)     │   │
│  │ - rollout.py         │   │   KL(N_ref || N_new)          │   │
│  │ - recording/         │   └───────────────────────────────┘   │
│  │ - distributed.py    │                                        │
│  │ - seed.py            │                                        │
│  └──────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DiffSBDD (Read-Only Reference)                    │
│  ConditionalDDPM / EGNNDynamics / LigandPocketDDPM              │
└─────────────────────────────────────────────────────────────────┘
```

## 训练管线

```
1. build_runtime_ddp()
   → model_impl = DiffSBDDModel(...)
   → ref_model = copy.deepcopy(model_impl)      # KL 参考模型
   → model_train = DDP(model_impl)               # DDP 包裹
   → actor = EDMActor(model_train, ..., ref_model=ref_model)

2. 训练循环 (main_diffsbdd_ddp.py)
   → rollout: DiffSBDDModel.sample() → 扩散采样
   → reward:  CompositeReward.calculate_rewards() → UniDock + QED + SA
   → filter:   diffsbdd_filter → 有效性过滤
   → update:   EDMActor.calculate_loss() → PPO clip + kl_coef × KL

3. KL 计算
   → current_model(**kwargs) → (log_prob, mu_new, sigma_new)
   → ref_model(**kwargs) → (mu_ref, sigma_ref)     # torch.no_grad()
   → compute_gaussian_kl(mu_ref, sigma_ref, mu_new, sigma_new, mask)
```

## 目录结构

```
DiffB110/
├── ddpo/                              # 核心集成代码
│   ├── main_diffsbdd_ddp.py           # ★ 多卡 DDP 入口
│   ├── main_diffsbdd_single_gpu.py    # 单卡调试入口
│   ├── main_diffsbdd.py               # 旧入口（legacy）
│   ├── integration/                   # DiffSBDD 适配
│   │   ├── diffsbdd_model.py          # DiffSBDDModel
│   │   └── mask_utils.py              # mask 转换
│   ├── kl/                            # KL 散度约束
│   │   └── gaussian_kl.py             # compute_gaussian_kl
│   ├── rewards/                       # 奖励系统
│   │   ├── docking.py                 # 对接奖励主逻辑
│   │   ├── docking_preparation.py     # SDF 准备 + 加氢 + Gasteiger
│   │   ├── docking_config.py          # 配置常量
│   │   ├── docking_registry.py        # 后端注册
│   │   ├── docking_factory.py         # 工厂函数
│   │   ├── docking_backends/          # 3 个后端
│   │   ├── interfaces/                # 抽象接口
│   │   ├── normalization.py           # z-score 归一化
│   │   ├── composite.py               # 组合奖励
│   │   ├── qed_reward.py              # QED
│   │   ├── sa_reward.py               # SA
│   │   └── stability.py               # 稳定性
│   ├── runtime/                       # 运行时
│   │   ├── build.py                  # ★ 工厂：YAML → 全部组件
│   │   ├── rollout.py                 # 轨迹采样
│   │   ├── recording/                 # 训练记录
│   │   ├── data_loader.py             # 数据加载
│   │   ├── distributed.py            # DDP 通信
│   │   ├── seed.py                    # 种子管理
│   │   └── train_helpers.py           # 训练辅助
│   ├── utils/                         # 工具
│   │   ├── molecule_semantics.py      # ★ 统一分子管线
│   │   ├── diffsbdd_filter.py         # 有效性过滤
│   │   ├── molecule_recorder.py       # 分子记录
│   │   ├── batch_recorder.py          # batch 统计
│   │   └── logger.py                  # 日志
│   └── patches/                       # 历史补丁
├── DiffSBDD/                          # 原始扩散模型（READ-ONLY）
├── RLPF-verl_diffusion/               # RL 框架参考
│   └── verl_diffusion/worker/actor/edm_actor.py  # EDMActor
├── configs/                           # YAML 配置
│   └── qed_sa_docking_8gjl_8gpu.yaml  # ★ 当前主配置
├── data/                              # 受体/配体/map
├── scripts/                           # 启动脚本
│   └── run_rl_qed_sa_docking_8gjl_8gpu.sh  # ★ 当前 8 卡启动
├── test/                              # 测试
├── materials/                         # 设计笔记（流形/判别器方案）
└── OUT/                               # 训练输出
```

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 多卡训练 | torchrun + DDP | Lightning 已弃用，直接 DDP 更可控 |
| 对接后端 | UniDock 为主 | GPU 加速，8GJL 配置已验证 |
| 分子管线 | OpenBabel→RDKit 单一管线 | 避免语义分歧，所有下游共享同一 rdmol |
| Docking 输入 | 加氢 + Gasteiger 电荷 | 提升对接精度，从 canonical mol 一次性准备 |
| KL 约束 | deepcopy ref_model | 独立参数，无共享，torch.no_grad() 前向 |
| 奖励归一化 | 全局 z-score | 废弃 group_index，统一 RewardNormalizer |
| 组件构建 | runtime/build.py 工厂 | 配置驱动，YAML → 全部组件 |
