# BInD 单项目生成测试报告

## 结论

BInD 已完成独立环境、官方 checkpoint/test 数据下载、GPU 分子生成烟测。最小测试生成 1 个 test pocket 的 1 个分子，墙钟时间 `64.23 s`，其中 1000-step denoise 循环约 `42.97 s`。输出 SDF 已生成。

## 本地项目与环境

- 项目路径：`/data/disk6/DiffB110/reference project/pharmacophore/BInD`
- 独立环境：`bind_ref`
- Python：`3.9.23`
- PyTorch：`1.11.0`
- CUDA build：`11.3`
- PyG：`2.1.0`
- GPU 可见性：提权执行下 `torch.cuda.is_available=True`，`device_count=8`，设备 0 为 `NVIDIA GeForce RTX 3080`

环境安装过程中遇到一次老 PyTorch 与新版 MKL 的兼容错误：

```text
ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

处理方式：将 `bind_ref` 内 MKL 降级到 `<2024.1`，之后 PyTorch/torch_geometric/BInD 生成脚本导入正常。

## 下载文件

下载目录：`/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/bind_downloads`

| 文件 | 来源 | 大小/内容 |
| --- | --- | --- |
| `trained_checkpoint.tar.gz` | README 官方 Google Drive trained checkpoint | 10.7 MB |
| `processed_only_test.tar.gz` | README 官方 processed only test | 3.33 MB，100 个 `.pkl` test pocket |
| `data_split_keys.tar.gz` | README 官方 split keys | `split/sequence.pt` |
| `povme_data.tar.gz` | README 官方 POVME data | train/test CSV |

解压目录：`/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/bind_assets`

关键文件：

- `bind_assets/public/config.yaml`
- `bind_assets/public/model_best.pt`
- `bind_assets/crossdocked_sequence_split_onlytest/1.pkl`
- `bind_assets/povme/sequence_split_train.csv`
- `bind_assets/povme/sequence_split_test.csv`

## 测试配置

配置文件：`/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/bind_generate_test_min.yaml`

最小化项：

- `data_id: 1`
- `bs: 1`
- `n_generate: 1`
- `repaint_iter: 1`
- guidance 全部设为 `0.0`

说明：这是为了测试 checkpoint 到 SDF 的最短可运行路径与生成耗时；默认 README 配置会启用更重的 repaint/guidance，速度会更慢。

## 生成命令与时间

工作目录：

```bash
/data/disk6/DiffB110/reference project/pharmacophore/BInD
```

命令：

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/bind_mplconfig timeout 900 /usr/bin/time -p conda run -n bind_ref python generate_test_pockets.py /data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/bind_generate_test_min.yaml
```

结果：

```text
All keys matched successfully
1000/1000 denoise steps completed
Generated SMILES: NCc1ccc2c(c1O)C2(CO)C(O)NC(=O)C1=CNC=NC1
real 64.23
user 54.05
sys 8.35
```

## 输出

输出目录：`/data/disk6/DiffB110/analysis/reference_project_generation_20260530_173617/bind_outputs/min_test`

生成文件：

- `gen/gen_1.sdf`：2012 bytes
- `gen/gen_1.pkl`：1822 bytes
- `log.log`
- `config.yaml`

## 注意事项

- 没有修改 BInD 原始源码、原始配置或 DiffB110 训练文件。
- 这次安装覆盖了生成测试所需的核心依赖；README 中 docking/posecheck/pose evaluation 相关依赖没有用于本次 `generate_test_pockets.py`，未作为生成速度测试前置条件。
- BInD 代码中生成路径硬编码 `.cuda()`，普通沙箱下 CUDA 不可见；真实生成测速必须使用 GPU 可见的提权执行。
