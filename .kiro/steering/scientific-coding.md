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
- Prefer minimal, local, reversible code changes.
- Preserve existing APIs unless change is necessary.
- Follow the existing project style.
- Keep code explicit and debuggable.
- Do not fabricate results, metrics, docking scores, or molecule validity.

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
