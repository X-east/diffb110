# launch archive

> 最后更新：2026-06-18

这里保存不再作为 `launch/` 根目录活跃入口的脚本、配置、生成矩阵、日志和 cache。

## 子目录

| 目录 | 内容 | 默认是否作为新入口 |
|---|---|---|
| `legacy_diffsbdd_wrappers/` | 旧 6/8GPU wrapper 和被 `launch/run.sh` 取代的配置 | 否 |
| `diffgui_historical/` | DiffGui 单次历史实验脚本、旧矩阵和模板配置 | 否 |
| `research_lines/` | anti-carbon guard、edge diffusion、x/h/e、MolSkill 等研究线配置 | 否 |
| `generated/` | 自动生成矩阵 YAML、matrix CSV/JSON | 否 |
| `logs/` | 矩阵 launcher 日志 | 否 |
| `runtime_logs/` | 原先散落在 `launch/` 根目录的 nohup 日志 | 否 |
| `cache/` | Python bytecode cache | 否 |

## 使用原则

新任务优先使用 `launch/README.md` 里的活跃入口。archive 下的脚本只用于复现历史实验；若需要重新运行，先检查其中硬编码路径和依赖环境是否仍然有效。
