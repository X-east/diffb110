from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RewardStats:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float


def _as_tensor(values: Any, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(device=device, dtype=torch.float32).reshape(-1)
    else:
        tensor = torch.tensor(list(values), device=device, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    finite_mask = torch.isfinite(tensor)
    return tensor[finite_mask]


def compute_global_reward_stats(values: Any, dist_ctx: Any, epsilon: float = 1.0e-6) -> RewardStats:
    """Compute mean/std/min/max over one local reward vector.

    Uses all-reduce when native PyTorch DDP is enabled, otherwise falls back to
    local statistics. The returned std is floored by ``epsilon``.
    """
    device = getattr(dist_ctx, "device", torch.device("cpu"))
    local = _as_tensor(values, device=device)

    if local.numel() == 0:
        count = torch.tensor(0.0, device=device, dtype=torch.float64)
        sum_ = torch.tensor(0.0, device=device, dtype=torch.float64)
        sq_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        min_ = torch.tensor(float("inf"), device=device, dtype=torch.float64)
        max_ = torch.tensor(float("-inf"), device=device, dtype=torch.float64)
    else:
        local64 = local.to(dtype=torch.float64)
        count = torch.tensor(float(local64.numel()), device=device, dtype=torch.float64)
        sum_ = local64.sum()
        sq_sum = (local64 * local64).sum()
        min_ = local64.min()
        max_ = local64.max()

    if getattr(dist_ctx, "enabled", False) and dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_, op=dist.ReduceOp.SUM)
        dist.all_reduce(sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(min_, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_, op=dist.ReduceOp.MAX)

    global_count = int(count.item())
    if global_count <= 0:
        return RewardStats(count=0, mean=0.0, std=1.0, minimum=0.0, maximum=0.0)

    mean = float((sum_ / count).item())
    variance = float((sq_sum / count).item()) - mean * mean
    variance = max(variance, 0.0)
    std = max(variance ** 0.5, float(epsilon))
    minimum = float(min_.item())
    maximum = float(max_.item())
    return RewardStats(count=global_count, mean=mean, std=std, minimum=minimum, maximum=maximum)
