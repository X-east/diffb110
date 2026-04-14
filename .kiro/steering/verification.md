# 任务验证规范

## 验证脚本位置

所有任务验证代码统一写入: `scripts/verify_completed_tasks.py`

## 验证规则

### 1. 子任务验证 (每一小步)

每完成一个子任务 (如 4.1, 4.2, 4.3)，必须:

1. 在 `verify_completed_tasks.py` 中添加对应的验证函数
2. 验证函数命名: `check_<功能描述>()`
3. 使用 `run_check(name, fn)` 记录结果
4. 验证内容包括:
   - 文件是否存在
   - 类/函数是否可导入
   - 方法签名是否正确
   - 基本功能是否正常

### 2. 大任务验证 (每一大步)

每完成一个大任务 (如 Task 4, Task 5)，必须:

1. 创建 `verify_task_N()` 函数汇总所有子任务验证
2. 按子任务编号分组打印验证结果
3. 运行相关的属性测试 (如果有)

### 3. 验证脚本结构

```python
# Task N: 任务名称
def verify_task_N():
    print("\n[N.1] 子任务描述")
    run_check("检查项名称", check_function)
    
    print("\n[N.2] 子任务描述")
    run_check("检查项名称", check_function)
```

### 4. 使用方法

```bash
# 验证所有已完成任务
python scripts/verify_completed_tasks.py

# 只验证特定任务
python scripts/verify_completed_tasks.py --task N

# 同时运行属性测试
python scripts/verify_completed_tasks.py --run-tests
```

### 5. 验证时机

| 时机 | 操作 |
|------|------|
| 完成子任务 | 添加验证代码，运行 `--task N` |
| 完成大任务 | 运行完整验证 + `--run-tests` |
| Checkpoint | 运行所有验证，确保全部通过 |

### 6. 验证内容对照表

| Task | 子任务 | 验证内容 |
|------|--------|---------|
| 1 | 1.1-1.4 | 目录结构、配置文件、BaseReward |
| 2 | 2.1-2.2 | mask_utils 函数、Property 1 |
| 3 | 3.1 | PDB 解析、DiffSBDD 模块 |
| 4 | 4.1-4.8 | DiffSBDDModel 各方法 |
| 5 | - | Checkpoint: 所有测试通过 |
| 6 | 6.1-6.2 | molecule_converter、Property 13 |
| 7 | 7.1 | receptor PDBQT 文件 |
| 8 | 8.1-8.3 | DockingReward、Property 4, 9 |
| 9 | 9.1-9.2 | DruglikenessReward、Property 5, 6 |
| 10 | 10.1-10.2 | CompositeReward、Property 7 |
| 11 | 11.1-11.2 | DiffSBDDFilter、Property 8, 15 |
| 12 | - | Checkpoint: 奖励系统测试通过 |
| 13 | 13.1-13.3 | main_diffsbdd.py 入口 |
| 14 | 14.1 | 端到端集成测试 |
| 15 | - | Final Checkpoint |

## 文档时间戳规范

**所有生成的说明文档必须包含时间戳**

### 时间戳格式

在文档开头添加：
```markdown
# 文档标题

**创建时间**: YYYY-MM-DD HH:MM:SS
**最后更新**: YYYY-MM-DD HH:MM:SS
```

### 适用文档类型

- 测试结果报告（如 `*_TEST_RESULTS.md`）
- 配置指南（如 `*_GUIDE.md`）
- 实现计划（如 `*_PLAN.md`）
- 分析报告（如 `*_ANALYSIS.md`）
- 总结文档（如 `*_SUMMARY.md`）

### 示例

```markdown
# GPU RL 测试结果

**创建时间**: 2026-02-28 10:30:00
**最后更新**: 2026-02-28 14:45:00

## 测试环境
...
```

## 注意事项

- 验证代码应该能独立运行，不依赖 GPU 或 checkpoint 文件
- 对于需要模型实例的测试，使用 mock 或跳过
- 保持验证脚本简洁，避免重复代码
- 验证失败时打印清晰的错误信息
- **所有说明文档必须包含创建和更新时间戳**
