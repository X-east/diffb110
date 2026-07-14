"""Tests for topology-aware DiffGui CPU affinity assignment."""

from __future__ import annotations

import pytest

from ddpo.diffgui.runtime.cpu_affinity import select_rank_cpu_assignments


def _dual_thread_core_groups() -> list[tuple[int, ...]]:
    """Return the 16-core/32-thread topology used by the training host."""
    return [(core, core + 16) for core in range(16)]


def test_seven_ranks_receive_four_non_overlapping_logical_cpus() -> None:
    """Seven ranks should consume 28 threads and leave two physical cores free."""
    assignments = select_rank_cpu_assignments(
        _dual_thread_core_groups(),
        world_size=7,
        logical_cpus_per_rank=4,
    )

    assert assignments == [
        (0, 1, 16, 17),
        (2, 3, 18, 19),
        (4, 5, 20, 21),
        (6, 7, 22, 23),
        (8, 9, 24, 25),
        (10, 11, 26, 27),
        (12, 13, 28, 29),
    ]
    flattened = [cpu for assignment in assignments for cpu in assignment]
    assert len(flattened) == len(set(flattened)) == 28
    assert set(range(32)) - set(flattened) == {14, 15, 30, 31}


def test_assignment_rejects_insufficient_cpu_capacity() -> None:
    """Affinity configuration should fail before training when CPUs are insufficient."""
    with pytest.raises(RuntimeError, match="Insufficient CPU affinity capacity"):
        select_rank_cpu_assignments(
            _dual_thread_core_groups()[:4],
            world_size=3,
            logical_cpus_per_rank=4,
        )


def test_assignment_preserves_complete_sibling_groups() -> None:
    """A rank allocation must not split one physical core's sibling threads."""
    with pytest.raises(RuntimeError, match="preserve complete sibling groups"):
        select_rank_cpu_assignments(
            _dual_thread_core_groups(),
            world_size=2,
            logical_cpus_per_rank=3,
        )
