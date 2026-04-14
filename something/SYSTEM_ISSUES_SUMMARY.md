# DiffB110 项目 - 系统问题诊断与解决方案

## 🔴 检测到的系统问题

### 1. **GPU 驱动与 CUDA 版本不匹配**
```
PyTorch CUDA: 11.8
系统 CUDA: 12.0
结果: GPU 无法被 PyTorch 检测到
```

### 2. **系统权限配置损坏**
```
/etc/sudo.conf 所有者错误
/etc/sudoers 无法访问
结果: 无法使用 sudo 提升权限
```

### 3. **Conda 多进程权限问题**
```
multiprocessing.SemLock 权限被拒绝
结果: 无法创建新环境
```

---

## ✅ 解决方案

### 方案 A: 使用现有 dif 环境（推荐 - 最快）

虽然 GPU 检测有问题，但驱动已加载。尝试直接运行训练：

```bash
conda activate dif
cd /home/md666/data/disk6/DiffB110

# 使用 CPU 模式运行（临时测试）
python ddpo/main_diffsbdd_lightning.py \
  --config_path ddpo/configs/ddp_8gpu_production.yaml \
  --device cpu
```

### 方案 B: 修复 PyTorch CUDA 版本（需要权限）

如果能获得 root 权限：

```bash
# 1. 修复 sudo 配置
sudo dpkg-reconfigure sudo

# 2. 重新安装 PyTorch for CUDA 12.1
conda activate dif
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 验证
python -c "import torch; print(torch.cuda.is_available())"
```

### 方案 C: 使用 Docker（最可靠）

创建 Dockerfile：

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git

WORKDIR /workspace
COPY . /workspace

RUN pip install pytorch-lightning ray[tune] wandb tensordict \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

CMD ["python", "ddpo/main_diffsbdd_lightning.py", \
     "--config_path", "ddpo/configs/ddp_8gpu_production.yaml"]
```

构建和运行：
```bash
docker build -t diffb110-gpu .
docker run --gpus all -v $(pwd):/workspace diffb110-gpu
```

### 方案 D: 在 CPU 上运行（用于测试）

修改配置文件 `ddp_8gpu_production.yaml`：

```yaml
model:
  device: "cpu"  # 改为 cpu

ddp:
  enabled: false  # 禁用 DDP
  num_gpus: 0
```

然后运行：
```bash
conda activate dif
cd /home/md666/data/disk6/DiffB110
python ddpo/main_diffsbdd_lightning.py \
  --config_path ddpo/configs/ddp_8gpu_production.yaml
```

---

## 🔍 系统信息

### GPU 硬件
```
✓ 8x NVIDIA RTX 3080 GPU 检测到
✓ NVIDIA 驱动 590.48.01 已加载
✓ CUDA 工具链 12.0 已安装
```

### 软件环境
```
✓ PyTorch 2.0.1 已安装 (CUDA 11.8 版本)
✗ CUDA 版本不匹配 (需要 12.0/12.1)
✗ 系统权限配置损坏
```

---

## 🚀 推荐步骤

### 立即可行（无需权限）
1. 在 CPU 上运行测试
2. 验证代码和配置正确
3. 等待系统管理员修复权限

### 需要权限
1. 修复 sudo 配置
2. 重新安装 PyTorch for CUDA 12.1
3. 验证 GPU 检测

### 最可靠方案
1. 使用 Docker 运行
2. 完全避免权限问题
3. 自动处理 CUDA 版本

---

## 📋 快速命令参考

```bash
# 检查 GPU 硬件
lspci | grep -i nvidia

# 检查驱动
lsmod | grep nvidia

# 检查 CUDA
nvcc --version

# 检查 PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 在 CPU 上运行
python ddpo/main_diffsbdd_lightning.py --device cpu

# 使用 Docker
docker run --gpus all -v $(pwd):/workspace diffb110-gpu
```

---

## 💡 建议

1. **短期**: 在 CPU 上运行测试，验证代码
2. **中期**: 联系系统管理员修复权限和 CUDA 版本
3. **长期**: 使用 Docker 进行生产训练

---

**系统有 8 个 RTX 3080 GPU，但需要修复配置才能使用。** 🎯
