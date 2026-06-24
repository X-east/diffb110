# TODO Cleanup

> **最后更新**: 2026-06-24

## 已完成

- `README.md` 已同步为 2026-06-18 的项目顶层说明。
- `docs/project_context.md` 已补充 DiffGui RL、D-MPNN、双 R 路线结论和证据规则。
- `docs/project_paths.md` 已同步当前路径和 `output/` 非活跃状态。
- `docs/workflow_commands.md` 已补充 DiffGui 环境、D-MPNN、分析记录和当前入口。
- `materials/README.md` 已新增，明确 `materials/` 是历史设计笔记 archive。
- `.cursor/rules/` 与 `.codebuddy/rules/` 保持同步，不删除。
- 2026-06-24：主线表述已从 DiffSBDD RL 切换为 DiffGui RL，优先同步 `AGENTS.md`、`README.md`、`docs/project_context.md`、`docs/workflow_commands.md`、`docs/project_paths.md` 和 `launch/README.md`。

## 待确认

### 0. 主线切换后的旧口径残留

当前主线已经切到 DiffGui 强化学习。后续如继续清理，应优先确认：

- `.cursor/rules/` 与 `.codebuddy/rules/` 是否仍有旧 DiffSBDD 主线、旧 Lightning、旧 `something/` 输出规则；
- 是否需要把 `docs/源数据_DiffSBDD-xhe大改线最终结果.md` 移到 `materials/` 或 work record archive；
- README / docs / launch README 中是否还需要补充最新 DiffGui 正式训练结果链接。

### 1. `output/` 历史目录

当前 `output/` 不是活跃输出根目录，但本地仍存在：

```text
output/pdf/260601-260605 每日总结合并.pdf
```

建议后续确认是否迁移到：

- `work_days/每周/`
- 对应 `work_records/`
- 或删除历史副本

在确认前，不把 `output/` 作为新运行或新分析输出目录。

### 2. PyTorch version compatibility warning

当前主环境历史记录为 `torch==2.0.1+cu118`，部分组件曾提示期望 `PyTorch >= 2.2`。

需要在真正遇到兼容性错误时再评估是否升级；不要为了消除 warning 破坏 DiffSBDD/DiffGui 现有环境。

### 3. 大型输出和分析目录的 git 边界

用户已要求 git 提交时忽略 `analysis/` 相关大型产物。后续提交前应确认：

- 不添加 `OUT/`
- 不添加大型 SDF/模型/benchmark 输出
- 不添加 `output/`
- 只添加本次明确需要进入版本控制的源码、配置、测试和文档
