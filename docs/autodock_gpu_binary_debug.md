# AutoDock-GPU Binary 查找失败问题排查文档

**日期**: 2026-03-15  
**问题**: 8GPU DDP 训练中，所有进程都报 `AutoDock-GPU binary not found`，导致 `reward_docking=-1.0000`

---
2026-03-15 09:53:03,733 - DiffSBDD-DDPO - INFO - [E0 B0] n=16 valid=4 (25%) | reward mean=-0.8072  std=0.3466  min=-1.0000  max=-0.1251  |  reward_docking=-1.0000  reward_druglikeness=-0.6144  score_qed=0.1111  score_sa=0.1600
/home/md666/miniforge3/envs/dif/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py:84: UserWarning: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 16. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
  warning_cache.warn(
    /data/disk6/DiffB110/DiffSBDD/analysis/molecule_builder.py:184: UserWarning: Sanitization failed. Returning None.
  warnings.warn('Sanitization failed. Returning None.')
[DockingReward/AutoDockGPU] Built 15/16 valid SDF files
/data/disk6/DiffB110/ddpo/rewards/docking.py:588: UserWarning: [DockingReward/AutoDockGPU] dock_batch failed (gpu_id=0): AutoDock-GPU binary not found.
Options:
  1. conda install -c conda-forge autodock-gpu
  2. Download from https://github.com/ccsb-scripps/AutoDock-GPU/releases
  3. Set env var AUTODOCK_GPU_BIN=/path/to/autodock_gpu_64wi
Traceback (most recent call last):
  File "/data/disk6/DiffB110/ddpo/rewards/docking.py", line 581, in _dock_batch_autodock_gpu
    backend = self._get_autodock_backend()
  File "/data/disk6/DiffB110/ddpo/rewards/docking.py", line 510, in _get_autodock_backend
    self._autodock_backend = AutoDockGPUBackend(
  File "/data/disk6/DiffB110/ddpo/rewards/docking_backends/autodock_gpu.py", line 121, in __init__
    raise RuntimeError(
RuntimeError: AutoDock-GPU binary not found.
Options:
  1. conda install -c conda-forge autodock-gpu
  2. Download from https://github.com/ccsb-scripps/AutoDock-GPU/releases
  3. Set env var AUTODOCK_GPU_BIN=/path/to/autodock_gpu_64wi

  warnings.warn(
[DockingReward] Done: mean=-1.0000  min=-1.0000  max=-1.0000  valid(non-penalty)=0/16
## 一、环境信息

| 项目 | 路径/值 |
|------|--------|
| Binary 实际路径 | `/home/md666/data/disk6/DiffB110/AutoDock-GPU/AutoDock-GPU-develop/bin/autodock_gpu_64wi` |
| 系统软链接 | `/usr/local/bin/autodock_gpu_64wi` → 上述路径 |
| 软链接权限 | `lrwxrwxrwx 1 root root` (root 创建) |
| `/usr/local/bin` 是否在 PATH | 是 (`echo $PATH` 确认) |
| `which autodock_gpu_64wi` | **无输出**（conda 环境激活后 PATH 被重排） |
| conda 环境 | `dif` |
| 训练启动方式 | `torchrun --nproc_per_node=8` |

---

## 二、问题根因分析

### 根因 1：模块导入时 binary 查找（已修复）

`autodock_gpu.py` 模块级别代码：
```python
AUTODOCK_GPU_BIN: Optional[str] = _find_autodock_gpu_bin()  # 导入时执行
```

`torchrun` 子进程在 `import` 时执行 `_find_autodock_gpu_bin()`，此时：
- `AUTODOCK_GPU_BIN` 环境变量尚未设置（`export` 在脚本里 `torchrun` 之前，但模块可能已被其他方式预导入）
- 或者 `shutil.which` 在子进程的 PATH 里找不到 binary

**修复**：将模块级常量改为 `None`，在 `__init__` 里调用 `_find_autodock_gpu_bin()`（懒加载）。

### 根因 2：torchrun 子进程 PATH 与主 shell 不同

`conda activate dif` 只在 interactive shell 里修改 PATH。`torchrun` 用 `subprocess` 启动子进程，PATH 可能不包含 conda 的 bin 目录，导致 `/usr/local/bin` 也可能被排序到后面或根本找不到软链接。

**修复尝试**：在 `run_8gpu_training.sh` 里 `export AUTODOCK_GPU_BIN=<绝对路径>` — 但**仍然失败**，原因见根因 1（模块导入时机早于 export 生效）。

### 根因 3（当前状态，仍在排查）

即使修复了懒加载，`09:51:44` 的日志仍然报 `binary not found`。

可能原因：
1. **`_find_autodock_gpu_bin()` 里 `os.environ.get("AUTODOCK_GPU_BIN")` 读取的是子进程自己的环境**，而 `export` 在父进程（bash 脚本）里设置，子进程应该能继承 —— 理论上没问题
2. **binary 文件本身权限问题**：`autodock_gpu_64wi` 可能没有执行权限，或者软链接目标不可访问
3. **`Path(env_bin).exists()` 检查失败**：文件路径正确但 `exists()` 返回 False（例如文件系统挂载问题，disk6 是网络/外部磁盘）

---

## 三、已尝试的解决方案

| 方案 | 状态 | 结果 |
|------|------|------|
| 在 bash 脚本里 `export AUTODOCK_GPU_BIN=<绝对路径>` | ✅ 已实施 | 仍然失败 |
| 将 `AUTODOCK_GPU_BIN` 改为懒加载（`__init__` 里调用） | ✅ 已实施 | 仍然失败 |
| 删除 `CUDA_VISIBLE_DEVICES` 设置（之前会破坏 `--devnum` 映射） | ✅ 已实施 | 解决了部分问题 |
| 缓存 `AutoDockGPUBackend` 实例（避免每 batch 重新初始化） | ✅ 已实施 | 减少重复初始化 |

---

## 四、下一步排查方向

### 方向 A：直接在代码里硬编码路径（最可靠）

**提议**：在 `_find_autodock_gpu_bin()` 里加一个硬编码的 fallback 路径：
```python
_HARDCODED_BIN = "/home/md666/data/disk6/DiffB110/AutoDock-GPU/AutoDock-GPU-develop/bin/autodock_gpu_64wi"

def _find_autodock_gpu_bin():
    env_bin = os.environ.get("AUTODOCK_GPU_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    for name in _CANDIDATE_BINS:
        found = shutil.which(name)
        if found:
            return found
    # Hardcoded fallback for this installation
    if Path(_HARDCODED_BIN).exists():
        return _HARDCODED_BIN
    return None
```

### 方向 B：验证文件系统可访问性

在训练脚本里加验证：
```bash
echo "Binary check:"
ls -la /home/md666/data/disk6/DiffB110/AutoDock-GPU/AutoDock-GPU-develop/bin/autodock_gpu_64wi
python -c "from pathlib import Path; print(Path('/home/md666/data/disk6/DiffB110/AutoDock-GPU/AutoDock-GPU-develop/bin/autodock_gpu_64wi').exists())"
```

### 方向 C：直接传参给 DockingReward

在 `main_diffsbdd_lightning.py` 里初始化 `DockingReward` 时显式传入 `autodock_gpu_bin` 参数，完全绕过 binary 查找逻辑。

---

## 五、关键代码位置

| 文件 | 关键位置 |
|------|----------|
| `ddpo/rewards/docking_backends/autodock_gpu.py` | `_find_autodock_gpu_bin()`, `AutoDockGPUBackend.__init__` |
| `ddpo/rewards/docking.py` | `DockingReward.__init__`, `_get_autodock_backend()`, `_dock_batch_autodock_gpu()` |
| `ddpo/trainer/lightning_ddpo_module.py` | `setup()`, `_update_docking_gpu_id()` |
| `ddpo/main_diffsbdd_lightning.py` | `DockingReward(...)` 初始化 |
| `run_8gpu_training.sh` | `export AUTODOCK_GPU_BIN=...`, `torchrun` 命令 |
| `run_4gpu_test.sh` | 同上 |
