# SDF 对接结果生成 2D 排名图脚本

## 脚本

`make_docking_2d_previews.py`

## 当前功能（用户最终选择：第一版）

- 从去重并按 docking score 排序的实验 SDF 读取候选分子。
- 从单独的重对接对照 SDF 读取参考分子。
- 生成 16:9 白底页面，参考分子固定在右上角，下方为 5×5 候选网格；全部候选按每页 25 个自动分页。
- 所有候选和参考分子均标注 docking score。
- 生成 `mol001` 等展示名及可追溯 mapping CSV。
- 只清理 `STEREOANY/EITHERDOUBLE` 等不确定双键绘图标记，保留明确 E/Z 与手性。
- 原始 SDF 只读，不写回 2D 坐标。

## 当前输入配置

脚本顶部的 `ROOT` 指向去重 SDF 结果目录，`SYSTEMS` 定义四个体系的实验 SDF、参考 SDF、参考名称和预期记录数。

后续同类任务直接复制并修改 `ROOT`/`SYSTEMS` 配置即可，不需要重写绘图、布局、化学清理和映射逻辑。

## 运行环境与命令

```powershell
conda run --no-capture-output -n PhoreGen python make_docking_2d_previews.py
```

当前 PhoreGen 环境的 RDKit 与 NumPy 2 会打印 ABI 警告，但本次绘图成功完成。若未来环境升级，优先使用 RDKit/NumPy 兼容的环境。

## 本次固定参数

- 页面：6000×3375 内部画布，输出预览 3600×2025 PNG。
- 候选：第一页前 25 个。
- 网格：5 列×5 行。
- 参考分子：右上角。
- 多重键偏移：`0.20`。
- 名称：`mol001`、`mol002`……
- 排序：docking score 数值升序，越负越好。
- 最终采用第一版字号：候选名称 42、候选分数 28；参考名称 34、参考分数 25（基于 6000×3375 内部画布）。
- 最终采用第一版分子尺寸与间距：结构最大使用约 94% 单元格宽度和 67% 高度。
- 同时输出 `pages_highres`（6000×3375）和 `pages_preview`（3600×2025）。
