from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def is_distributed_enabled(config: dict[str, Any]) -> bool:
    """Return whether native PyTorch DDP is enabled in config."""
    return bool(config.get("ddp", {}).get("enabled", False))


def init_distributed_from_env(config: dict[str, Any]) -> DistributedContext:
    """Initialize torch.distributed from torchrun-style environment variables.

    When DDP is disabled, returns a single-process context without touching
    torch.distributed state.
    """
    if not is_distributed_enabled(config):
        if torch.cuda.is_available():
            device = torch.device(config.get("model", {}).get("device", "cuda"))
        else:
            device = torch.device("cpu")
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
        )

    required = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "DDP is enabled but required distributed environment variables are missing: "
            f"{missing}. Launch with torchrun or provide the torch.distributed env vars."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    configured_num_gpus = config.get("ddp", {}).get("num_gpus")
    if configured_num_gpus is not None and int(configured_num_gpus) != world_size:
        raise RuntimeError(
            "DDP world size does not match config.ddp.num_gpus: "
            f"WORLD_SIZE={world_size}, ddp.num_gpus={configured_num_gpus}. "
            "Launch torchrun with --nproc_per_node matching ddp.num_gpus, "
            "or update the config."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("DDP is enabled but CUDA is not available.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def destroy_distributed() -> None:
    """Destroy the active process group when initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(ctx: DistributedContext) -> bool:
    """Return whether the current process is rank zero."""
    return int(ctx.rank) == 0


def barrier_if_needed(ctx: DistributedContext) -> None:
    """Synchronize ranks only when DDP is enabled."""
    if ctx.enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: Any, ctx: DistributedContext, src: int = 0) -> Any:
    """Broadcast a small Python object from src to all ranks."""
    if not ctx.enabled:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_gather_objects(obj: Any, ctx: DistributedContext) -> list[Any]:
    """Gather a small Python object from every rank."""
    if not ctx.enabled:
        return [obj]
    output = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(output, obj)
    return output


def _reduce_scalar(value: float, ctx: DistributedContext, op: dist.ReduceOp) -> float:
    if not ctx.enabled:
        return float(value)
    tensor = torch.tensor(float(value), device=ctx.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def reduce_sum_scalar(value: float, ctx: DistributedContext) -> float:
    """All-reduce one scalar with SUM semantics."""
    return _reduce_scalar(value, ctx, dist.ReduceOp.SUM)


def reduce_mean_scalar(value: float, ctx: DistributedContext) -> float:
    """All-reduce one scalar with MEAN semantics."""
    reduced = reduce_sum_scalar(value, ctx)
    return reduced / float(ctx.world_size)


def reduce_min_scalar(value: float, ctx: DistributedContext) -> float:
    """All-reduce one scalar with MIN semantics."""
    return _reduce_scalar(value, ctx, dist.ReduceOp.MIN)


def reduce_max_scalar(value: float, ctx: DistributedContext) -> float:
    """All-reduce one scalar with MAX semantics."""
    return _reduce_scalar(value, ctx, dist.ReduceOp.MAX)
