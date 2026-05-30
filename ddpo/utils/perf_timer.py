"""
PerfTimer: lightweight step-level wall-clock profiler.

Records elapsed time for named pipeline stages using time.perf_counter().
Supports manual start/stop and context-manager APIs.
"""

from __future__ import annotations

import time
from typing import Dict


class _TimerContext:
    """Context manager returned by PerfTimer.measure()."""

    def __init__(self, timer: PerfTimer, step_name: str) -> None:
        self._timer = timer
        self._step_name = step_name

    def __enter__(self) -> _TimerContext:
        self._timer.start(self._step_name)
        return self

    def __exit__(self, *args) -> None:
        self._timer.stop()


class PerfTimer:
    """High-precision step timer for profiling training pipeline stages.

    Usage::

        timer = PerfTimer()
        with timer.measure("generate"):
            ...
        timer.start("reward_qed")
        ...
        timer.stop()
        print(timer)   # formatted summary
        print(timer.summary())   # dict[str, float]
    """

    def __init__(self) -> None:
        self._timings: Dict[str, float] = {}
        self._current_step: str | None = None
        self._current_start: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(self, step_name: str) -> _TimerContext:
        """Context manager that times the enclosed block.

        Example::

            with timer.measure("generate"):
                samples = rollout.generate_samples(prompts)
        """
        return _TimerContext(self, step_name)

    def start(self, step_name: str) -> None:
        """Begin timing *step_name*.  Finalises any previously-started step."""
        self._finalize()
        self._current_step = step_name
        self._current_start = time.perf_counter()

    def stop(self) -> float:
        """Finalise the current step and return its elapsed time (seconds)."""
        if self._current_step is not None and self._current_start is not None:
            elapsed = time.perf_counter() - self._current_start
            self._timings[self._current_step] = self._timings.get(self._current_step, 0.0) + elapsed
            step = self._current_step
            self._current_step = None
            self._current_start = None
            return elapsed
        return 0.0

    def get(self, step_name: str) -> float:
        """Retrieve recorded time for *step_name*, or 0.0."""
        return self._timings.get(step_name, 0.0)

    def summary(self) -> Dict[str, float]:
        """Return a copy of all recorded timings."""
        return dict(self._timings)

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._finalize()
        self._timings.clear()

    def __repr__(self) -> str:
        if not self._timings:
            return "PerfTimer(<empty>)"
        lines: list[str] = []
        total = 0.0
        for step, elapsed in self._timings.items():
            lines.append(f"  {step:30s}: {elapsed:7.3f}s")
            total += elapsed
        lines.append(f"  {'─' * 40}")
        lines.append(f"  {'total':30s}: {total:7.3f}s")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        if self._current_step is not None:
            self.stop()
