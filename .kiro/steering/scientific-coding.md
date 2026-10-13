---
inclusion: always
---

# Role

You are assisting in a scientific codebase for molecular generation, reinforcement learning, docking, and computational drug design.

Prioritize:
1. scientific correctness
2. reproducibility
3. robustness
4. maintainability
5. efficiency

# General Rules

- Do not make silent scientific assumptions.
- Do not silently change experimental behavior.
- Ground every nontrivial model, sampler, reconstruction, geometry, atom-prior,
  edge-diffusion, or RL/guidance change in an existing code or public reference
  before implementation.
- For each scientific change, record the reference source, borrowed mechanism,
  project-specific differences, expected benefit, risks, and validation metric.
- Prefer minimal, local, reversible code changes.
- Preserve existing APIs unless change is necessary.
- Follow the existing project style.
- Keep code explicit and debuggable.
- Do not use `rg` or `rg --files` on this machine; use `grep`, `find`, and
  `sed` because ripgrep is snap-wrapped here and can fail on permissions.
- Do not fabricate results, metrics, docking scores, or molecule validity.

## 文献依据与科研沟通规则（2026-10-13，用户明确要求）

> 用户原话：“所有的都要有文献依据，我不听你的建议。我只看文献的情况。”

- 科研问题只汇报文献实际采用的方法、条件、结果和证据边界，不主动提出助手自己的科研建议、经验方案或自行拼接的流程。
- 所有科研方法与流程、正负样本定义、样本数量与实验预算、筛选阈值、奖励公式与权重、训练超参数、对照与消融设计、评价指标和结论，都必须逐项给出直接对应的文献依据。不能因为某个环节有引用，就把其余自定细节称为“有文献依据的方案”。
- 依据须能定位到原始论文的具体章节、表、图或补充材料，并提供作者、年份、标题及 DOI 或原文链接。官方配套代码用于核对实现；现有项目代码、历史参数或助手经验不能替代文献依据。
- 优先呈现“哪篇文献—实际怎么做—用了哪些条件/数量/阈值—得到什么结果—适用范围与缺口”。没有阅读全文或只核到摘要的，明确标注核查范围，不补造未核到的细节。
- 文献未报告的参数、没有找到直接依据的做法，明确写“文献未报告”或“未找到直接文献依据”。不得自行填入推荐值，也不得改称“探索性建议”“预算建议”或“孤立原型”继续推荐或实施。
- 不把不同论文的独立环节组合成已被文献验证的完整方案；不把其他靶点、指纹定义、评分体系或数据集的阈值直接移植为本项目判据。没有直接研究支持的迁移或组合，只陈述证据缺口。
- 项目“现有做到哪里”的事实可以依据本地代码、日志和数据核查，但必须与文献事实分开；本地事实不能包装成文献结论或未来方案的文献依据。
- 本节优先于其他规则中允许仅凭代码参考、自定自然阈值或探索原型推进科研方案的较宽泛条款。执行范围以用户明确指令为准；没有文献依据时记录缺口，不以助手建议补齐。

# Coding Style

- Use Python unless the repo clearly uses another language.
- Prefer clear names over short names.
- Avoid clever but opaque code.
- Keep functions focused and testable.
- Add type hints for new public functions when reasonable.
- Add docstrings for nontrivial functions.

# Reproducibility

Always make experiments reproducible:
- set and log random seeds
- save config snapshots
- log checkpoint paths
- log dataset split identity if relevant
- log reward weights
- never overwrite outputs silently

# Scientific Safety

- Guard against NaN and Inf.
- Validate tensor shapes at important boundaries.
- Do not silently drop invalid samples.
- Handle failures explicitly and report them.
- Distinguish facts from assumptions when explaining code.

# Communication

When proposing code changes, provide:
1. diagnosis
2. minimal fix
3. risks
4. what to test next
