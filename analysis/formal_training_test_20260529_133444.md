# 7GPU 正式训练启动测试记录

## 测试背景

本次测试用于确认当前未提交的显存分阶段 instrumentation、rollout CPU 暂存、`rollout_accum_steps=4` 和正式 7GPU 配置能否在真实训练路径中启动并进入 reward / actor 阶段。此记录是阶段性运行记录，不是完整 25 epoch 训练总结。

## 启动信息

| 项目 | 值 |
|---|---|
| 记录时间 | 2026-05-29 13:45:51 CST，13:48:11 补充 PPO1 状态 |
| 配置 | `launch/7gpu_dock_classifier_v0_1.yaml` |
| 启动命令 | `CONFIG_PATH=launch/7gpu_dock_classifier_v0_1.yaml LOG_LEVEL=INFO nohup bash launch/run.sh > analysis/formal_training_test_20260529_133444_launch.log 2>&1 &` |
| 外层 PID | `2940171` |
| torchrun PID | `2940209` |
| rank PIDs | `2940279`-`2940285` |
| 输出目录 | `OUT/7gpu-dock-classifier-v0-1_20260529_133609` |
| 启动日志 | `analysis/formal_training_test_20260529_133444_launch.log` |
| rank-0 训练日志 | `OUT/7gpu-dock-classifier-v0-1_20260529_133609/logs/rank_00/logs/training_20260529_133611.log` |

## 配置摘要

| 配置项 | 值 |
|---|---:|
| DDP GPUs | 7 |
| GPU IDs | `0,1,2,3,4,5,6` |
| per-rank batch | 64 |
| global batch | 448 |
| rollout accum steps | 4 |
| effective generated molecules / epoch | 1792 |
| train micro batch size | 16 |
| PPO epochs | 4 |
| configured epochs | 25 |
| reward weights | docking 0.5, chembl_classifier 0.5 |

## 已验证阶段

| 阶段 | 结果 |
|---|---|
| 7GPU DDP 启动 | 通过，rank 0-6 均启动 |
| reward 构建 | 通过，启用 `docking` 和 `chembl_classifier` |
| rollout accum | 通过，epoch 1 完成 4/4 个 rollout chunk |
| rollout CPU 暂存 | 通过，每个 chunk 的 DataProto tensor 在 CPU |
| concat 后 batch | 通过，rank-0 DataProto tensor 总量约 `cpu=273.8MB` |
| reward 阶段 | 通过，reward 计算完成并进入 RewardNormalizer |
| actor 阶段 | 已进入，首个 PPO microbatch 已 materialize 并释放 |
| PPO 内循环 | 已进入 `ppo1`，说明 `ppo0` 已完成 |
| 完整 epoch summary | 记录时尚未完成 |

## 关键日志摘录

| 时间 | 事件 | 关键数值 |
|---|---|---|
| 13:36:14 | 开始 25 epoch DDP 训练 | world_size=7, per_rank_batch=64, global_batch=448 |
| 13:39:11 | rollout chunk concat 完成 | DataProto tensors total `cpu=273.8MB` |
| 13:39:45 | RewardNormalizer 完成 | raw_mean=0.4260, raw_std=0.5935, failures=75 |
| 13:39:45 | prepare 后显存 | allocated=20.4MB, reserved=3424.0MB, max_allocated=2931.4MB |
| 13:40:17 | actor 首个 microbatch 释放后 | allocated=32.4MB, reserved=5026.0MB, max_allocated=4719.2MB |
| 13:48:11 | 进入 PPO1 首个 microbatch | allocated=53.3MB, reserved=7324.0MB, max_allocated=4903.4MB |

## GPU 状态快照

13:48 左右查询到 GPU 0-6 均处于正式训练占用中：

| GPU | 显存使用 | 总显存 | 利用率 |
|---:|---:|---:|---:|
| 0 | 7697 MiB | 10240 MiB | 81% |
| 1 | 7023 MiB | 10240 MiB | 93% |
| 2 | 5491 MiB | 10240 MiB | 94% |
| 3 | 5553 MiB | 10240 MiB | 91% |
| 4 | 7055 MiB | 10240 MiB | 82% |
| 5 | 7035 MiB | 10240 MiB | 91% |
| 6 | 6737 MiB | 10240 MiB | 93% |

GPU 7 不在本次 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` 中，查询时仍有约 1501 MiB 其它占用。

## 阶段性判断

1. 正式训练不是只在 smoke 路径可用，当前配置已经跑过真实 7GPU DDP、4 段 rollout、docking/classifier reward 和 PPO actor 首个 microbatch。
2. `rollout_accum_steps=4` 的缓存压力主要转移到了 CPU：rank-0 concat 后约 273.8MB，按 7 卡粗略估算全局 rollout tensor 在 CPU 侧约 1.9GB 量级。
3. GPU 峰值主要来自采样和 actor backward，而不是 rollout batch 长期驻留。rank-0 采样阶段 max_allocated 约 2.93GB，actor 阶段已观察到 max_allocated 约 4.90GB、reserved 约 7.32GB。
4. 当前 10GB 卡仍有余量，但正式 epoch 尚未完整结束，后续仍需观察完整 PPO 4 epoch、checkpoint 和 epoch summary 阶段。

## 未完成/需继续观察

| 项目 | 状态 |
|---|---|
| Epoch 1 完整结束 | 记录时未完成 |
| `training_summary.csv` 首行 epoch 指标 | 记录时尚未写入 |
| checkpoint 写入 | 需到 configured checkpoint interval |
| 长时间稳定性 | 需继续观察至少 1 个完整 epoch |
| 旧的 `bash launch/run.sh` / defunct `torchrun` 进程 | 已观察到历史残留 PID，但未处理 |
