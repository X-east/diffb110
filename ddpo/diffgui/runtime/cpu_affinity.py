"""Topology-aware CPU affinity for DiffGui DDP ranks and subprocesses."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_rank_cpu_affinity(
    config: dict[str, Any],
    *,
    local_rank: int,
    world_size: int,
    logger_obj: logging.Logger | None = None,
) -> dict[str, Any]:
    """Bind one DDP rank and its child processes to configured logical CPUs."""
    cfg = ((config.get("diffgui_runtime", {}) or {}).get("cpu_affinity", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return {"enabled": 0, "local_rank": int(local_rank), "logical_cpus": []}

    logical_cpus_per_rank = int(cfg.get("logical_cpus_per_rank", 4))
    if logical_cpus_per_rank <= 0:
        raise ValueError("diffgui_runtime.cpu_affinity.logical_cpus_per_rank must be positive")
    available_cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    core_groups = _read_core_groups(available_cpus)
    assignments = select_rank_cpu_assignments(
        core_groups,
        world_size=int(world_size),
        logical_cpus_per_rank=logical_cpus_per_rank,
    )
    selected_cpus = assignments[int(local_rank)]
    _set_all_process_threads_affinity(selected_cpus)
    for key in _THREAD_ENV_KEYS:
        os.environ[key] = str(logical_cpus_per_rank)

    payload = {
        "enabled": 1,
        "local_rank": int(local_rank),
        "world_size": int(world_size),
        "logical_cpus_per_rank": int(logical_cpus_per_rank),
        "logical_cpus": list(selected_cpus),
        "available_logical_cpus": available_cpus,
    }
    (logger_obj or logging.getLogger(__name__)).info("cpu_affinity=%s", payload)
    return payload


def select_rank_cpu_assignments(
    core_groups: list[tuple[int, ...]],
    *,
    world_size: int,
    logical_cpus_per_rank: int,
) -> list[tuple[int, ...]]:
    """Assign complete sibling groups to ranks without cross-rank overlap."""
    if world_size <= 0 or logical_cpus_per_rank <= 0:
        raise ValueError("world_size and logical_cpus_per_rank must be positive")
    group_sizes = {len(group) for group in core_groups if group}
    if len(group_sizes) > 1:
        raise RuntimeError(f"Non-uniform CPU sibling groups are unsupported: sizes={sorted(group_sizes)}")
    sibling_count = next(iter(group_sizes), 1)
    if int(logical_cpus_per_rank) % int(sibling_count) != 0:
        raise RuntimeError(
            "logical_cpus_per_rank must preserve complete sibling groups "
            f"logical_cpus_per_rank={logical_cpus_per_rank} siblings_per_core={sibling_count}"
        )
    flattened = [int(cpu) for group in core_groups for cpu in group]
    required = int(world_size) * int(logical_cpus_per_rank)
    if len(flattened) < required:
        raise RuntimeError(
            "Insufficient CPU affinity capacity "
            f"available_logical_cpus={len(flattened)} required={required}"
        )
    return [
        tuple(sorted(flattened[rank * logical_cpus_per_rank : (rank + 1) * logical_cpus_per_rank]))
        for rank in range(int(world_size))
    ]


def current_cpu_affinity() -> list[int]:
    """Return the current thread's logical CPU affinity."""
    return sorted(int(cpu) for cpu in os.sched_getaffinity(0))


def _read_core_groups(available_cpus: list[int]) -> list[tuple[int, ...]]:
    """Group available logical CPUs by physical package and core id."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cpu in available_cpus:
        topology_dir = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package_id = int((topology_dir / "physical_package_id").read_text(encoding="utf-8").strip())
            core_id = int((topology_dir / "core_id").read_text(encoding="utf-8").strip())
        except (FileNotFoundError, OSError, ValueError):
            package_id, core_id = 0, int(cpu)
        groups[(package_id, core_id)].append(int(cpu))
    return [tuple(sorted(groups[key])) for key in sorted(groups)]


def _set_all_process_threads_affinity(cpus: tuple[int, ...]) -> None:
    """Apply one affinity mask to every currently existing process thread."""
    cpu_set = set(int(cpu) for cpu in cpus)
    task_dir = Path("/proc/self/task")
    task_ids = [int(path.name) for path in task_dir.iterdir() if path.name.isdigit()]
    for task_id in task_ids:
        try:
            os.sched_setaffinity(task_id, cpu_set)
        except ProcessLookupError:
            continue
    os.sched_setaffinity(0, cpu_set)
