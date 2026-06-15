# PoLiGenX 10 分子生成测试报告

## 结论

- 项目路径：`reference project/pharmacophore/PoLiGenX`
- 本轮状态：未能进行真实生成测速。
- 主要原因：本地没有 PoLiGenX 生成模型 checkpoint，README 明确写明模型权重需邮件索取。
- 源码状态：未修改 PoLiGenX 源码、配置或数据目录。

## 本地资源检查

已检查权重/输入相关文件：

```bash
find 'reference project/pharmacophore/PoLiGenX' -maxdepth 4 \
  -type f \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \)
```

结果：未发现可用于 `experiments/generate_ligands.py` / `generate_ligands_multi.py` 的模型权重。

README 的 Sampling 章节写明：

```text
Currently, we provide the model weights upon request. Please contact us via email.
```

因此当前没有可自动下载的公开 checkpoint 链接。

## 环境与生成测试状态

- PoLiGenX 环境文件存在：
  - `environment.yml`
  - `environment_vina.yml`
  - `environment_walters.yml`
- 本机尚未建立独立 PoLiGenX 环境。
- 因为缺少生成 checkpoint，即使建立环境也无法完成用户要求的“下载 checkpoint 并生成 10 个分子”。

## 对 DiffB110 的判断

PoLiGenX 在方法上值得关注：它是 pocket-conditioned equivariant diffusion，包含 latent conditioning、shape constraints、SA importance sampling 等设计。但在当前本机条件下，它不能形成可复现速度基准。要继续测试，需要先拿到官方或用户提供的 `best_valid.ckpt`/等价 checkpoint，然后再建立独立环境并运行 10 分子生成。
