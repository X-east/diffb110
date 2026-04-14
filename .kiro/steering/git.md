# Git 版本控制规范

## 核心规则

**每完成一个子任务（如 Task 4.1、4.2、4.3），必须立即执行以下操作：**

1. **提交代码**
2. **打标签**

## 提交流程

```bash
# 1. 添加相关文件
git add <修改的文件>

# 2. 提交（使用规范的 commit message）
git commit -m "feat(module): description (Task X.Y)"

# 3. 打标签
git tag -a task-X.Y-complete -m "Task X.Y: description"
```

## Commit Message 格式

```
<type>(<scope>): <description> (Task X.Y)

类型 (type):
- feat: 新功能
- fix: 修复 bug
- docs: 文档更新
- test: 测试相关
- refactor: 重构
- chore: 杂项

示例:
- feat(model): implement get_mask method (Task 4.2)
- feat(reward): implement DockingReward (Task 8.1)
- test(model): add property test for log_prob (Task 4.5)
```

## 标签命名规范

```
task-X.Y-complete

示例:
- task-4.1-complete
- task-4.2-complete
- task-8.1-complete
```

## 检查命令

```bash
# 查看所有标签
git tag -l

# 查看最近提交
git log --oneline -10

# 回退到某个子任务
git checkout task-4.1-complete
```

## 为什么这样做？

1. **精确回退**: 可以回退到任意子任务状态
2. **进度追踪**: 清晰看到每个子任务的完成时间
3. **问题定位**: 出问题时快速定位是哪个子任务引入的
4. **协作友好**: 团队成员可以看到详细的开发历史

## 注意事项

- 不要等到大任务完成才提交，每个子任务完成就提交
- 标签一旦打上不要删除，保持历史完整性
- 如果忘记打标签，可以后补：`git tag -a task-X.Y-complete <commit-hash> -m "..."`
