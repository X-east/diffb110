# AutoDock-GPU Binary 查找失败问题排查文档

**日期**: 2026-03-15  
**状态**: 已归档（当前主后端已切换为 UniDock）

> **注意**: 当前 8GJL 8卡配置使用 **UniDock** 作为主对接后端。AutoDock-GPU 仍作为备选后端保留。本文档记录的是 2026-03-15 的 AutoDock-GPU 排查历史，供参考。

---

## 问题摘要

8GPU DDP 训练中，所有进程都报 `AutoDock-GPU binary not found`，导致 `reward_docking=-1.0000`。

## 环境信息

| 项目 | 路径/值 |
|------|--------|
| Binary 实际路径 | `/home/md666/data/disk6/DiffB110/AutoDock-GPU/AutoDock-GPU-develop/bin/autodock_gpu_64wi` |
| 系统软链接 | `/usr/local/bin/autodock_gpu_64wi` → 上述路径 |
| conda 环境 | `dif` |
| 训练启动方式 | `torchrun --nproc_per_node=8` |

## 根因分析

### 根因 1：模块导入时 binary 查找
`autodock_gpu.py` 模块级别代码在 import 时执行 `_find_autodock_gpu_bin()`，此时环境变量可能未生效。

**修复**：将模块级常量改为 `None`，在 `__init__` 里调用（懒加载）。已实施。

### 根因 2：torchrun 子进程 PATH 与主 shell 不同
`conda activate dif` 只在 interactive shell 里修改 PATH，`torchrun` 子进程的 PATH 可能不包含所需路径。

### 根因 3：文件系统可访问性
`/data/disk6` 是外部磁盘，`Path.exists()` 可能因文件系统挂载问题返回 False。

## 已实施的解决方案

| 方案 | 状态 | 结果 |
|------|------|------|
| bash 脚本里 `export AUTODOCK_GPU_BIN=<绝对路径>` | ✅ | 仍失败 |
| 懒加载（`__init__` 里调用） | ✅ | 减少问题 |
| 删除 `CUDA_VISIBLE_DEVICES` 设置 | ✅ | 解决了部分问题 |
| 缓存 `AutoDockGPUBackend` 实例 | ✅ | 减少重复初始化 |

## 当前状态

- **当前主后端**: UniDock（8GJL 配置），不再受此问题影响
- **AutoDock-GPU**: 作为备选后端保留，使用时需确保 binary 可被找到
- **相关代码**: `ddpo/rewards/docking_backends/autodock_gpu.py`

## 关键代码位置

| 文件 | 关键位置 |
|------|----------|
| `ddpo/rewards/docking_backends/autodock_gpu.py` | `_find_autodock_gpu_bin()`, `AutoDockGPUBackend.__init__` |
| `ddpo/rewards/docking.py` | `DockingReward.__init__`, 后端选择逻辑 |
| `ddpo/rewards/docking_registry.py` | 后端注册与选择 |
| `ddpo/rewards/docking_config.py` | 后端名常量 |
