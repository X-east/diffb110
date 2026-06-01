# PILOT 生成测试报告

## 项目与环境

- 项目路径：`reference project/pharmacophore/PILOT`
- 官方 remote：`https://github.com/jule-c/PILOT.git`
- 目标独立环境：`pilot_ref`
- 环境文件：`reference project/pharmacophore/PILOT/environment.yml`
- 本轮未修改 PILOT 代码、配置或数据目录。

## 环境安装尝试

运行命令：

```bash
conda env create -n pilot_ref -f environment.yml
```

结果：

- conda 依赖解析成功。
- 下载阶段失败。
- 失败包：`pytorch-2.0.1`，大小约 1.41 GB。
- 报错核心：`Connection broken: IncompleteRead(27717120 bytes read, 1483062338 more expected)`。
- `conda env list` 中未出现 `pilot_ref`。
- `/home/md666/miniforge3/envs/pilot_ref` 未创建成功。

## Checkpoint 状态

- 本地搜索 `*.ckpt/*.pt/*.pth`：未找到模型 checkpoint。
- README 的 Sampling 部分只写了 “Model checkpoints go here!”。
- README 未给出可直接下载的 checkpoint URL、release、Zenodo/HuggingFace 链接或文件名。
- 因此本轮没有可执行的 checkpoint 下载命令。

## 生成测试状态

- 生成命令未运行。
- 原因 1：`pilot_ref` 独立环境未安装成功。
- 原因 2：本地没有 PILOT checkpoint，README 也没有可下载 checkpoint 链接。

## 当前结论

PILOT 当前卡在两个前置条件：

1. 独立环境 `pilot_ref` 因大包下载中断未建成。
2. checkpoint 来源缺失，无法加载模型做分子生成。

因此本项目本轮没有生成分子，也没有有效生成耗时。后续要继续，需要先解决 PyTorch 大包下载稳定性，或提供可用的 PILOT checkpoint 路径/下载链接。
