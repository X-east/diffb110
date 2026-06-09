# 任务验证规范

**最后更新**: 2026-05-08

## 已完成阶段

原始 Task 1-15 已全部完成。项目经历了大规模解耦重构，当前状态已远超原始 Task 规划。

### 已完成关键里程碑

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| DiffSBDD 适配 | ✅ | DiffSBDDModel + mask_utils |
| 口袋加载 | ✅ | 8GJJ / 8GJL 受体 |
| 奖励系统 | ✅ | UniDock + QED + SA，4层解耦 |
| DDP 多卡训练 | ✅ | torchrun + main_diffsbdd_ddp.py |
| 配置驱动 | ✅ | runtime/build.py 工厂 |
| KL 散度约束 | ✅ 已接入 | kl_coef 可配，默认 0.0（未实际测试） |
| 分子语义统一 | ✅ | molecule_semantics.py + prepare_mol_for_docking |
| 加氢+Gasteiger | ✅ | 从 canonical mol 一次性准备 |
| 奖励归一化 | ✅ | z-score RewardNormalizer，group_index 已废弃 |
| 训练记录系统 | ✅ | recording/ 模块 |

---

## 当前验证清单

### P0：训练正确性

| 验证项 | 状态 | 说明 |
|--------|------|------|
| KL NaN/Inf 防护 | ❌ 未修复 | `compute_gaussian_kl()` 缺少 clamp + NaN guard |
| KL 分离日志 | ❌ 未实现 | `calculate_loss()` 不返回 kl_penalty |
| KL 启用测试 | ❌ 未测试 | kl_coef=0.0，需先修复上面两项 |
| 完整训练流程验证 | ❌ 未完成 | 8卡 DDP 端到端指标追踪 |
| 加氢+Gasteiger 对 docking 效果 | ❌ 未验证 | 真实训练中的对比评估 |

### P1：奖励系统验证

| 验证项 | 状态 | 说明 |
|--------|------|------|
| reward drift 检测 | ❌ 未实现 | 训练中分子质量退化监控 |
| multi-objective balance | ❌ 未验证 | QED/SA/docking 权重平衡 |

### P2：奖励漂移治理

| 验证项 | 状态 | 说明 |
|--------|------|------|
| 3D 几何正常性惩罚 | ❌ 未实现 | 异常键长/clash/UFF 能量 |
| descriptor Mahalanobis 惩罚 | ❌ 未实现 | 需要离线构建参考集 |
| 判别器正常性奖励 | ❌ 未实现 | 可选，见 materials/ |

---

## 验证脚本位置

所有任务验证代码统一写入: `scripts/verify_completed_tasks.py`

## 验证规则

### 子任务验证

每完成一个子任务，必须:
1. 在验证脚本中添加对应的验证函数
2. 验证函数命名: `check_<功能描述>()`
3. 验证内容包括: 文件是否存在、类/函数是否可导入、方法签名是否正确、基本功能是否正常

### 大任务验证

每完成一个大任务，必须:
1. 创建 `verify_task_N()` 函数汇总所有子任务验证
2. 运行相关的属性测试

### 使用方法

```bash
# 验证所有已完成任务
python scripts/verify_completed_tasks.py

# 只验证特定任务
python scripts/verify_completed_tasks.py --task N
```

## 文档时间戳规范

**所有生成的说明文档必须包含时间戳**

格式：
```markdown
# 文档标题

**创建时间**: YYYY-MM-DD HH:MM:SS
**最后更新**: YYYY-MM-DD HH:MM:SS
```

## 注意事项

- 验证代码应该能独立运行，不依赖 GPU 或 checkpoint 文件
- 对于需要模型实例的测试，使用 mock 或跳过
- 保持验证脚本简洁，避免重复代码
- 验证失败时打印清晰的错误信息

## 正式训练日志与指标记录

- 正式训练必须默认使用本地记录，不得依赖 WandB 或其他远程平台作为唯一证据来源。
- 每个正式 run 必须保存标量指标文件，例如 `metrics/metrics.csv`，并保留 stdout/stderr `.log`。
- 终端输出必须能看清训练状态：epoch、global step、elapsed time、总 loss，以及可用的分量 loss/acc。
- 模型改造训练应优先记录可解释分量，例如 base nll、edge loss、bond length loss、edge acc、real-bond acc、train/val loss。
- WandB 和其他远程实验平台默认禁用；除非用户明确要求一次性例外，否则正式训练不要启用。即便远程镜像被允许，本地指标也必须完整；如果本地指标缺失，必须先补齐或声明限制，再做质量结论。
