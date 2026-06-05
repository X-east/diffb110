---
inclusion: always
---
# Scope

This project uses diffusion-based molecular generation and reinforcement learning for optimization.

Protect:
- trajectory semantics
- reward semantics
- numerical stability
- mask correctness
- reproducibility

## Code Change Confirmation Rule

For this project, **execute code changes directly** without asking for confirmation first.
Always describe what was changed and why, but do not wait for user reply before making the edit.

Exception: changes with high risk of data loss, irreversible side effects, or that alter reward/diffusion/mask semantics should still be described first and flagged explicitly.

# Diffusion Rules

- Clearly distinguish x0, xt, noise, predicted noise, timestep, and trajectory state.
- Never silently reverse timestep order.
- Do not rename diffusion variables ambiguously in the same scope.
- Document whether a function does forward diffusion, reverse denoising, rollout, or log-prob evaluation.
- Preserve tensor shape semantics explicitly.

# Mask and Batch Rules

- Treat mask logic as critical.
- Never mix padded and valid nodes carelessly.
- Validate shapes for coordinates, features, masks, batch indices.
- Make padded-to-flat and flat-to-padded conversions explicit.

# RL Rules

- Never treat reward code as trivial.
- Keep reward components separate and inspectable.
- Preserve the distinction between: reward, penalty, advantage, old/new log prob, ratio, clipped ratio, entropy bonus.

# PPO / DDPO Rules

- Keep detach boundaries explicit.
- Keep reward scaling explicit.
- Keep advantage normalization explicit.
- Do not accidentally backpropagate through reward computation unless intended.
- Do not silently change clipping behavior.

# KL Divergence Rules

- KL constraint is computed as `KL[N(μ_ref, σ_ref²) || N(μ_new, σ_new²)]`
- `ref_model` is a frozen deepcopy — never update its parameters
- Always use `torch.no_grad()` when computing ref model forward pass
- `kl_coef` is configurable in YAML; currently defaults to 0.0
- **Known gaps** (fix before enabling kl_coef > 0):
  - `compute_gaussian_kl()` lacks NaN/Inf protection — extreme σ can cause explosion
  - No separate KL logging — cannot monitor KL contribution independently
- When enabling KL, monitor: KL magnitude, KL vs PPO loss ratio, KL trend over epochs

# Reward Design Rules

Composite rewards must:
- keep each component score visible
- keep weights in config
- log component means and failure counts
- separate invalid molecule failures from docking failures
- use `group_index` is **deprecated** — always use global z-score normalization via `RewardNormalizer`

Do not:
- hide hard-coded weights in code
- silently change reward meaning
- silently skip failed samples in evaluation

# Reward Drift Awareness

When training with RL, reward drift is a known risk:
- RL may optimize for docking score while degrading molecular quality
- Current mitigation: QED + SA multi-objective reward
- Future mitigations (see `materials/`):
  - 3D geometry normality penalty (abnormal bond lengths, clashes, UFF energy)
  - Descriptor Mahalanobis distance penalty (reference manifold constraint)
  - Optional discriminator normality reward

# Testing Priorities

For nontrivial changes, test:
- shape correctness
- mask correctness
- log-prob finiteness
- deterministic behavior under fixed seed where feasible
- one-batch sampling smoke test
- one-step reward smoke test
- one PPO/DDPO update smoke test
